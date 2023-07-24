import torch
import torch.nn.functional as F
from dgl.utils import gather_pinned_tensor_rows
import gs
from gs.utils import SeedGenerator, load_reddit, load_ogb, create_block_from_coo
import numpy as np
import time
import tqdm
import argparse
import pandas as pd
from model import *


def sample_w_o_relabel(P, seeds, seeds_ptr, fanouts):
    seedsts, units, colts, rowts, edatats = [], [], [], [], []
    num_batches = seeds_ptr.numel() - 1
    encoding_size = P._CAPI_get_num_rows()
    for fanout in fanouts:
        # (batchID * num_nodes) * nodeID
        subg, _ = P._CAPI_batch_slicing(seeds, seeds_ptr, 0, gs._CSC, gs._COO, False, True)
        probs = subg._CAPI_sum(1, 2, gs._COO)

        neighbors = torch.unique(subg._CAPI_get_coo_rows(False))
        # int(nodeID / num_nodes)
        node_probs = probs[neighbors]
        neighbors_ptr, _ = torch.ops.gs_ops.GetBatchOffsets(neighbors, num_batches, encoding_size)
        idx, _ = torch.ops.gs_ops.batch_list_sampling_with_probs(node_probs, fanout, False, neighbors_ptr)
        selected = neighbors[idx]

        nodes = torch.cat((subg._CAPI_get_cols(), selected)).unique()
        subg = subg._CAPI_slicing(nodes, 1, gs._COO, gs._COO, False)  # Row Slicing
        subg = subg._CAPI_divide(probs[nodes], 1, gs._COO)
        _sum = subg._CAPI_sum(0, 1, gs._COO)
        subg = subg._CAPI_divide(_sum, 0, gs._COO)

        encoded_coo_row = subg._CAPI_get_rows()[subg._CAPI_get_coo_rows(False)]
        # nodeID - int(nodeID / num_nodes) * num_nodes
        coo_ptr, coo_row = torch.ops.gs_ops.GetBatchOffsets(encoded_coo_row, num_batches, encoding_size)
        coo_col = seeds[subg._CAPI_get_coo_cols(False)]
        (
            unique_tensor,
            unique_tensor_ptr,
            sub_coo_row,
            sub_coo_col,
            sub_coo_ptr,
        ) = torch.ops.gs_ops.BatchCOORelabel(seeds, seeds_ptr, coo_col, coo_row, coo_ptr)
        seedst = torch.ops.gs_ops.SplitByOffset(seeds, seeds_ptr)
        unit = torch.ops.gs_ops.SplitByOffset(unique_tensor, unique_tensor_ptr)
        colt = torch.ops.gs_ops.SplitByOffset(sub_coo_col, sub_coo_ptr)
        rowt = torch.ops.gs_ops.SplitByOffset(sub_coo_row, sub_coo_ptr)
        edata = torch.ops.gs_ops.SplitByOffset(subg._CAPI_get_data("default"), sub_coo_ptr)
        seedsts.append(seedst)
        units.append(unit)
        colts.append(colt)
        rowts.append(rowt)
        edatats.append(edata)

        seeds, seeds_ptr = unique_tensor, unique_tensor_ptr
    return seedsts, units, colts, rowts, edatats


def compute_acc(pred, label):
    return (pred.argmax(1) == label).float().mean()


def train(dataset, args):
    device = args.device
    use_uva = args.use_uva
    fanouts = [int(x.strip()) for x in args.samples.split(",")]

    g, features, labels, n_classes, splitted_idx = dataset
    train_nid, val_nid = splitted_idx["train"], splitted_idx["valid"]
    g = g.to(device)
    weight = normalized_laplacian_edata(g)
    train_nid, val_nid = train_nid.to(device), val_nid.to(device)
    features, labels = features.to(device), labels.to(device)
    csc_indptr, csc_indices, edge_ids = g.adj_sparse("csc")
    weight = weight[edge_ids].to(device)
    if use_uva and device == "cpu":
        features, labels = features.pin_memory(), labels.pin_memory()
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
        train_nid, val_nid = train_nid.pin_memory(), val_nid.pin_memory()
    A = gs.Graph(False)
    A._CAPI_load_csc(csc_indptr, csc_indices)
    A._CAPI_set_data(weight)
    print("Check load successfully:", A._CAPI_metadata(), "\n")

    batch_size = 12800
    small_batch_size = args.batchsize
    num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)
    orig_seeds_ptr = torch.arange(num_batches + 1, dtype=torch.int64, device="cuda") * small_batch_size
    print(batch_size, small_batch_size, fanouts)

    train_seedloader = SeedGenerator(train_nid, batch_size=batch_size, shuffle=True, drop_last=False)
    val_seedloader = SeedGenerator(val_nid, batch_size=batch_size, shuffle=True, drop_last=False)
    model = SAGEModel(features.shape[1], 256, n_classes, len(fanouts)).to("cuda")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print("memory allocated before training:", static_memory / (1024 * 1024 * 1024), "GB")

    epoch_time = []
    cur_time = []
    acc_list = []
    start = time.time()
    for epoch in range(args.num_epoch):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tic = time.time()
        model.train()
        num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)
        for it, seeds in enumerate(tqdm.tqdm(train_seedloader)):
            seeds_ptr = orig_seeds_ptr
            if it == len(train_seedloader) - 1:
                num_batches = int((seeds.numel() + small_batch_size - 1) / small_batch_size)
                seeds_ptr = torch.arange(num_batches + 1, dtype=torch.int64, device="cuda") * small_batch_size
                seeds_ptr[-1] = seeds.numel()
            seeds, units, colts, rowts, edatats = sample_w_o_relabel(A, seeds, seeds_ptr, fanouts)

            for rank in range(num_batches):
                batch_seeds = seeds[0][rank]
                blocks = []
                for layer in range(len(fanouts)):
                    col_seed, unique, col, row, edata = (
                        seeds[layer][rank],
                        units[layer][rank],
                        colts[layer][rank],
                        rowts[layer][rank],
                        edatats[layer][rank],
                    )
                    block = create_block_from_coo(row, col, num_src=unique.numel(), num_dst=col_seed.numel())
                    block.edata["w"] = edata
                    block.srcdata["_ID"] = unique
                    blocks.insert(0, block)
                blocks = [block.to("cuda") for block in blocks]
                if use_uva:
                    batch_inputs = gather_pinned_tensor_rows(features, blocks[0].srcdata["_ID"])
                    batch_labels = gather_pinned_tensor_rows(labels, batch_seeds)
                else:
                    batch_inputs = features[blocks[0].srcdata["_ID"]].to("cuda")
                    batch_labels = labels[batch_seeds].to("cuda")

                batch_pred = model(blocks, batch_inputs)
                is_labeled = batch_labels == batch_labels
                batch_labels = batch_labels[is_labeled].long()
                batch_pred = batch_pred[is_labeled]
                loss = F.cross_entropy(batch_pred, batch_labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        val_pred = []
        val_labels = []
        with torch.no_grad():
            num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)
            for it, seeds in enumerate(tqdm.tqdm(val_seedloader)):
                seeds_ptr = orig_seeds_ptr
                if it == len(val_seedloader) - 1:
                    num_batches = int((seeds.numel() + small_batch_size - 1) / small_batch_size)
                    seeds_ptr = torch.arange(num_batches + 1, dtype=torch.int64, device="cuda") * small_batch_size
                    seeds_ptr[-1] = seeds.numel()
                seeds, units, colts, rowts, edatats = sample_w_o_relabel(A, seeds, seeds_ptr, fanouts)

                for rank in range(num_batches):
                    batch_seeds = seeds[0][rank]
                    blocks = []
                    for layer in range(len(fanouts)):
                        col_seed, unique, col, row, edata = (
                            seeds[layer][rank],
                            units[layer][rank],
                            colts[layer][rank],
                            rowts[layer][rank],
                            edatats[layer][rank],
                        )
                        block = create_block_from_coo(row, col, num_src=unique.numel(), num_dst=col_seed.numel())
                        block.edata["w"] = edata
                        block.srcdata["_ID"] = unique
                        blocks.insert(0, block)
                    blocks = [block.to("cuda") for block in blocks]
                    if use_uva:
                        batch_inputs = gather_pinned_tensor_rows(features, blocks[0].srcdata["_ID"])
                        batch_labels = gather_pinned_tensor_rows(labels, batch_seeds)
                    else:
                        batch_inputs = features[blocks[0].srcdata["_ID"]].to("cuda")
                        batch_labels = labels[batch_seeds].to("cuda")

                    batch_pred = model(blocks, batch_inputs)
                    is_labeled = batch_labels == batch_labels
                    batch_labels = batch_labels[is_labeled].long()
                    batch_pred = batch_pred[is_labeled]
                    val_pred.append(batch_pred)
                    val_labels.append(batch_labels)

        acc = compute_acc(torch.cat(val_pred, 0), torch.cat(val_labels, 0)).item()
        acc_list.append(acc)

        torch.cuda.synchronize()
        end = time.time()
        cur_time.append(end - start)
        epoch_time.append(end - tic)

        print("Epoch {:05d} | Val Acc {:.4f} | E2E Time {:.4f} s | Accumulated Time {:.4f} s".format(epoch, acc, epoch_time[-1], cur_time[-1]))

    torch.cuda.synchronize()
    total_time = time.time() - start

    print("Total Elapse Time:", total_time)
    print("Average Epoch Time:", np.mean(epoch_time[3:]))
    s5 = pd.Series(cur_time, name="cumulated time/s")
    s1 = pd.Series(acc_list, name="acc")
    s2 = pd.Series(epoch_time, name="time/s")
    s3 = pd.Series([total_time], name="total time/s")
    s4 = pd.Series([static_memory], name="static mem/GB")
    df = pd.concat([s5, s1, s2, s3, s4], axis=1)
    df.to_csv(
        "outputs/data/ladies_gsampler_{}_{}.csv".format(args.dataset, time.ctime().replace(" ", "_")),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Training model on gpu or cpu",
    )
    parser.add_argument(
        "--use-uva",
        type=bool,
        default=False,
        help="Wether to use UVA to sample graph and load feature",
    )
    parser.add_argument(
        "--dataset",
        default="products",
        choices=["reddit", "products", "papers100m"],
        help="which dataset to load for training",
    )
    parser.add_argument("--batchsize", type=int, default=512, help="batch size for training")
    parser.add_argument("--samples", default="4000,4000,4000", help="sample size for each layer")
    parser.add_argument("--num_epoch", type=int, default=100, help="numbers of epoch in training")
    args = parser.parse_args()
    print(args)

    if args.dataset == "reddit":
        dataset = load_reddit()
    elif args.dataset == "products":
        dataset = load_ogb("ogbn-products", "/home/ubuntu/dataset")
    elif args.dataset == "papers100m":
        dataset = load_ogb("ogbn-papers100M", "/home/ubuntu/dataset")
    print(dataset[0])
    train(dataset, args)
