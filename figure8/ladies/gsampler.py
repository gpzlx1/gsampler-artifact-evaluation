import gs
from gs.utils import SeedGenerator, load_graph
import torch
import torch.nn.functional as F
from dgl.utils import gather_pinned_tensor_rows
import numpy as np
import time
import tqdm
import argparse


def sample_w_o_relabel(P: gs.Matrix, fanouts, seeds, seeds_ptr):
    num_batches = seeds_ptr.numel() - 1
    graph = P._graph
    output_node = seeds
    blocks = []
    encoding_size = graph._CAPI_get_num_rows()
    for fanout in fanouts:
        # (batchID * num_nodes) * nodeID
        subg, _ = graph._CAPI_batch_slicing(seeds, seeds_ptr, 0, gs._CSC, gs._COO, False, True)
        probs = subg._CAPI_sum(1, 2, gs._CSR)

        neighbors = torch.unique(subg._CAPI_get_coo_rows(False))
        # int(nodeID / num_nodes)
        node_probs = probs[neighbors]
        neighbors_ptr, _ = torch.ops.gs_ops.GetBatchOffsets(neighbors, num_batches, encoding_size)
        idx, _ = torch.ops.gs_ops.batch_list_sampling_with_probs(node_probs, fanout, False, neighbors_ptr)
        selected = neighbors[idx]

        nodes = torch.cat((subg._CAPI_get_cols(), selected)).unique()
        subg = subg._CAPI_slicing(nodes, 1, gs._CSR, gs._COO, False)  # Row Slicing
        subg = subg._CAPI_divide(probs[nodes], 1, gs._COO)
        _sum = subg._CAPI_sum(0, 1, gs._CSC)
        subg = subg._CAPI_divide(_sum, 0, gs._COO)

        encoded_coo_row = subg._CAPI_get_rows()[subg._CAPI_get_coo_rows(False)]
        # nodeID - int(nodeID / num_nodes) * num_nodes
        coo_ptr, coo_row = torch.ops.gs_ops.GetBatchOffsets(encoded_coo_row, num_batches, encoding_size)
        coo_col = seeds[subg._CAPI_get_coo_cols(False)]
        unique_tensor, unique_tensor_ptr, sub_coo_row, sub_coo_col, sub_coo_ptr = torch.ops.gs_ops.BatchCOORelabel(
            seeds, seeds_ptr, coo_col, coo_row, coo_ptr
        )
        seedst = torch.ops.gs_ops.SplitByOffset(seeds, seeds_ptr)
        unit = torch.ops.gs_ops.SplitByOffset(unique_tensor, unique_tensor_ptr)
        colt = torch.ops.gs_ops.SplitByOffset(sub_coo_col, sub_coo_ptr)
        rowt = torch.ops.gs_ops.SplitByOffset(sub_coo_row, sub_coo_ptr)
        blocks.insert(0, (seedst, unit, colt, rowt))

        seeds, seeds_ptr = unique_tensor, unique_tensor_ptr
    input_node = seeds
    return input_node, output_node, blocks


def sample_w_relabel(P: gs.Matrix, fanouts, seeds, seeds_ptr):
    num_batches = seeds_ptr.numel() - 1
    graph = P._graph
    output_node = seeds
    blocks = []
    encoding_size = graph._CAPI_get_num_rows()
    for fanout in fanouts:
        subg, _ = graph._CAPI_batch_slicing(seeds, seeds_ptr, 0, gs._CSC, gs._COO, True, True)
        probs = subg._CAPI_sum(1, 2, gs._CSR)
        num_pick = np.min([probs.numel(), fanout])

        # int(nodeID / num_nodes)
        row_ptr, _ = torch.ops.gs_ops.GetBatchOffsets(subg._CAPI_get_rows(), num_batches, encoding_size)
        selected, _ = torch.ops.gs_ops.batch_list_sampling_with_probs(probs, num_pick, False, row_ptr)

        relabel_seeds_nodes = torch.ops.gs_ops.index_search(subg._CAPI_get_rows(), subg._CAPI_get_cols())
        nodes = torch.cat((relabel_seeds_nodes, selected)).unique()
        subg = subg._CAPI_slicing(nodes, 1, gs._CSR, gs._COO, False)  # Row Slicing
        subg = subg._CAPI_divide(probs[nodes], 1, gs._COO)
        _sum = subg._CAPI_sum(0, 1, gs._CSC)
        subg = subg._CAPI_divide(_sum, 0, gs._COO)

        encoded_coo_row = subg._CAPI_get_rows()[subg._CAPI_get_coo_rows(False)]
        # int(nodeID / num_nodes)
        coo_ptr, coo_row = torch.ops.gs_ops.GetBatchOffsets(encoded_coo_row, num_batches, encoding_size)
        coo_col = seeds[subg._CAPI_get_coo_cols(False)]
        unique_tensor, unique_tensor_ptr, sub_coo_row, sub_coo_col, sub_coo_ptr = torch.ops.gs_ops.BatchCOORelabel(
            seeds, seeds_ptr, coo_col, coo_row, coo_ptr
        )
        seedst = torch.ops.gs_ops.SplitByOffset(seeds, seeds_ptr)
        unit = torch.ops.gs_ops.SplitByOffset(unique_tensor, unique_tensor_ptr)
        colt = torch.ops.gs_ops.SplitByOffset(sub_coo_col, coo_ptr)
        rowt = torch.ops.gs_ops.SplitByOffset(sub_coo_row, coo_ptr)
        eweight = torch.ops.gs_ops.SplitByOffset(subg._CAPI_get_data("default"), coo_ptr)
        blocks.insert(0, (seedst, unit, colt, rowt, eweight))

        seeds, seeds_ptr = unique_tensor, unique_tensor_ptr
    input_node = seeds
    return input_node, output_node, blocks


def benchmark(args, matrix, nid, fanouts, n_epoch, sampler):
    print("####################################################{}".format(sampler.__name__))
    batch_size = args.batching_batchsize
    small_batch_size = args.batchsize
    num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)
    orig_seeds_ptr = torch.arange(num_batches + 1, dtype=torch.int64, device="cuda") * small_batch_size

    seedloader = SeedGenerator(nid, batch_size=batch_size, shuffle=True, drop_last=False)

    epoch_time = []
    mem_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print("memory allocated before training:", static_memory / (1024 * 1024 * 1024), "GB")
    for epoch in range(n_epoch):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        for it, seeds in enumerate(tqdm.tqdm(seedloader)):
            seeds_ptr = orig_seeds_ptr
            if it == len(seedloader) - 1:
                num_batches = int((seeds.numel() + small_batch_size - 1) / small_batch_size)
                seeds_ptr = torch.arange(num_batches + 1, dtype=torch.int64, device="cuda") * small_batch_size
                seeds_ptr[-1] = seeds.numel()
            input_nodes, output_nodes, blocks = sampler(matrix, fanouts, seeds, seeds_ptr)

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        mem_list.append((torch.cuda.max_memory_allocated() - static_memory) / (1024 * 1024 * 1024))

        print("Epoch {:05d} | Epoch Sample Time {:.4f} s | GPU Mem Peak {:.4f} GB".format(epoch, epoch_time[-1], mem_list[-1]))

    # use the first epoch to warm up
    print("Average epoch sampling time:", np.mean(epoch_time[1:]))
    print("Average epoch gpu mem peak:", np.mean(mem_list[1:]))
    print("####################################################END")


def train(dataset, args):
    device = args.device
    use_uva = args.use_uva
    fanouts = [int(x.strip()) for x in args.samples.split(",")]

    g, features, labels, n_classes, splitted_idx = dataset
    g = g.long()
    train_nid = splitted_idx["train"]
    g, train_nid = g.to(device), train_nid.to("cuda")
    # weight = normalized_laplacian_edata(g)
    weight = torch.ones(g.num_edges(), dtype=torch.float32, device=g.device)
    csc_indptr, csc_indices, edge_ids = g.adj_sparse("csc")
    weight = weight[edge_ids].to(device)
    if use_uva and device == "cpu":
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
        weight = weight.pin_memory()
    m = gs.Matrix(gs.Graph(False))
    m._graph._CAPI_load_csc(csc_indptr, csc_indices)
    m._graph._CAPI_set_data(weight)
    print("Check load successfully:", m._graph._CAPI_metadata(), "\n")

    n_epoch = 6
    if args.dataset in ("livejournal", "ogbn-products"):
        benchmark(args, m, train_nid, fanouts, n_epoch, sample_w_o_relabel)
    else:
        benchmark(args, m, train_nid, fanouts, n_epoch, sample_w_relabel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Training model on gpu or cpu")
    parser.add_argument("--use-uva", type=bool, default=False, help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset", default="ogbn-products", help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=512, help="batch size for training")
    parser.add_argument("--batching-batchsize", type=int, default=12800, help="batch size for training")
    parser.add_argument("--samples", default="4000,4000,4000", help="sample size for each layer")
    args = parser.parse_args()
    print(args)

    if args.dataset.startswith("ogbn"):
        dataset = load_graph.load_ogb(args.dataset, "/home/ubuntu/dataset")
    elif args.dataset == "livejournal":
        dataset = load_graph.load_dglgraph("/home/ubuntu/dataset/livejournal/livejournal.bin")
    elif args.dataset == "friendster":
        dataset = load_graph.load_dglgraph("/home/ubuntu/dataset/friendster/friendster.bin")
    else:
        raise NotImplementedError
    print(dataset[0])
    train(dataset, args)
