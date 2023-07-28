import gs
from gs.utils import SeedGenerator, load_graph
import torch
import torch.nn.functional as F
from dgl.utils import gather_pinned_tensor_rows
import numpy as np
import time
import tqdm
import argparse
import csv
from typing import List


def get_feature(features, rows, cols, use_uva):
    if use_uva:
        node_feats_u = gather_pinned_tensor_rows(features, rows)
        node_feats_v = gather_pinned_tensor_rows(features, cols)
    else:
        node_feats_u = features[rows]
        node_feats_v = features[cols]
    return node_feats_u, node_feats_v


torch.fx.wrap("get_feature")
torch.fx.wrap("gather_pinned_tensor_rows")


def asgcn_sampler(
    A: gs.Matrix,
    seeds: torch.Tensor,
    fanouts: List,
    features: torch.Tensor,
    W: torch.Tensor,
    use_uva: bool,
):
    output_nodes = seeds
    ret = []
    for K in fanouts:
        subA = A[:, seeds]
        p = subA.sum("w", axis=1).sqrt()
        node_feats_u, node_feats_v = get_feature(features, subA.rows(), subA.cols(), use_uva)
        h_u = node_feats_u @ W[:, 0]
        h_v = node_feats_v @ W[:, 1]
        h_v_sum = torch.sum(h_v)
        attention = torch.flatten((F.relu(h_u + h_v_sum) + 1) / K)
        g_u = torch.flatten(F.relu(h_u) + 1)
        q = F.normalize(p * attention * g_u, p=1.0, dim=0)

        sampleA, select_index = subA.collective_sampling(K, q, False)

        sampleA.edata["w"] = gs.ops.u_add_v(sampleA, h_u[select_index], h_v)
        sampleA.edata["w"] = (F.relu(sampleA.edata["w"]) + 1) / sampleA.num_rows()
        sampleA = sampleA.div("w", q[select_index], 1)

        seeds = sampleA.all_nodes()
        ret.append(sampleA)
    input_nodes = seeds
    return input_nodes, output_nodes, ret


def benchmark(args, graph, nid, fanouts, n_epoch, features, W, sampler):
    print("####################################################")
    seedloader = SeedGenerator(nid, batch_size=args.batchsize, shuffle=True, drop_last=False)

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
            input_nodes, output_nodes, blocks = sampler(graph, seeds, fanouts, features, W, args.use_uva)

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        mem_list.append((torch.cuda.max_memory_allocated() - static_memory) / (1024 * 1024 * 1024))

        print("Epoch {:05d} | Epoch Sample Time {:.4f} s | GPU Mem Peak {:.4f} GB".format(epoch, epoch_time[-1], mem_list[-1]))

    with open("outputs/result.csv", "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        # system name, dataset, sampling time, mem peak
        log_info = ["gSampler", args.dataset, np.mean(epoch_time[1:]), np.mean(mem_list[1:])]
        writer.writerow(log_info)

    # use the first epoch to warm up
    print("Average epoch sampling time:", np.mean(epoch_time[1:]))
    print("Average epoch gpu mem peak:", np.mean(mem_list[1:]))
    print("####################################################END")


def train(dataset, args):
    device = args.device
    fanouts = [int(x.strip()) for x in args.samples.split(",")]

    g, features, labels, n_classes, splitted_idx = dataset
    g = g.long().to(device)
    train_nid = splitted_idx["train"].to("cuda")
    # weight = normalized_laplacian_edata(g)
    weight = torch.ones(g.num_edges(), dtype=torch.float32, device=g.device)
    if features == None:
        features = torch.rand(g.num_nodes(), 128, dtype=torch.float32)
    features = features.to(device)
    W = torch.nn.init.xavier_normal_(torch.Tensor(features.shape[1], 2)).to("cuda")
    csc_indptr, csc_indices, edge_ids = g.adj_sparse("csc")
    weight = weight[edge_ids]
    if args.use_uva and device == "cpu":
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
        weight, features = weight.pin_memory(), features.pin_memory()
    m = gs.Matrix()
    m.load_graph("CSC", [csc_indptr, csc_indices])
    m.edata["w"] = weight

    rand_idx = torch.randint(0, train_nid.numel(), (args.batchsize,), device="cuda")
    seeds = train_nid[rand_idx]
    compile_func = gs.jit.compile(func=asgcn_sampler, args=(m, seeds, fanouts, features, W, args.use_uva))

    n_epoch = args.num_epoch
    benchmark(args, m, train_nid, fanouts, n_epoch, features, W, compile_func)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=6, help="run how many epochs")
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
    parser.add_argument("--dataset", default="ogbn-products", help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=256, help="batch size for training")
    parser.add_argument("--samples", default="512,512", help="sample size for each layer")
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
