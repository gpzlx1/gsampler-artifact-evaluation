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


def sample_w_o_relabel(A: gs.Matrix, seeds, fanouts, features, W_1, W_2, sample_a, use_uva):
    blocks = []
    output_nodes = seeds
    for fanout in fanouts:
        subg = A._graph._CAPI_slicing(seeds, 0, gs._CSC, gs._CSC + gs._COO, False)
        att3 = subg._CAPI_normalize(0, gs._CSC)._CAPI_get_data("default").unsqueeze(1)
        neighbors = torch.unique(subg._CAPI_get_coo_rows(False))
        subA = gs.Matrix(subg)
        if use_uva:
            u_feats = gather_pinned_tensor_rows(features, neighbors)
            v_feats = gather_pinned_tensor_rows(features, seeds)
        else:
            u_feats = features[neighbors]
            v_feats = features[seeds]
        u_feats_all_w1 = torch.empty((subg._CAPI_get_num_rows(), W_1.shape[1]), dtype=torch.float32, device="cuda")
        u_feats_all_w1[neighbors] = u_feats @ W_1
        v_feats_all_w1 = v_feats @ W_1
        u_feats_all_w2 = torch.empty((subg._CAPI_get_num_rows(), W_2.shape[1]), dtype=torch.float32, device="cuda")
        u_feats_all_w2[neighbors] = u_feats @ W_2
        v_feats_all_w2 = v_feats @ W_2

        res1 = gs.ops.u_mul_v(subA, u_feats_all_w1, v_feats_all_w1, gs._COO)
        att1 = torch.sum(res1, dim=1).unsqueeze(1)
        res2 = gs.ops.u_mul_v(subA, u_feats_all_w2, v_feats_all_w2, gs._COO)
        att2 = torch.sum(res2, dim=1).unsqueeze(1)
        att = torch.cat([att1, att2, att3], dim=1)
        att = F.relu(att @ F.softmax(sample_a, dim=0))
        att = att + 10e-10 * torch.ones_like(att)
        subA._graph._CAPI_set_data(att)
        subg = subA._graph._CAPI_sampling_with_probs(0, att, fanout, True, gs._CSC, gs._CSC)
        unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subg._CAPI_relabel()
        seeds = unique_tensor
    input_nodes = seeds
    return input_nodes, output_nodes, blocks


def sample_w_relabel(A: gs.Matrix, seeds, fanouts, features, W_1, W_2, sample_a, use_uva):
    blocks = []
    output_nodes = seeds
    for fanout in fanouts:
        subg = A._graph._CAPI_slicing(seeds, 0, gs._CSC, gs._CSC, True)
        rows = subg._CAPI_get_rows()
        subA = gs.Matrix(subg)
        if use_uva:
            u_feats = gather_pinned_tensor_rows(features, rows)
            v_feats = gather_pinned_tensor_rows(features, seeds)
        else:
            u_feats = features[rows]
            v_feats = features[seeds]

        res1 = gs.ops.u_mul_v(subA, u_feats @ W_1, v_feats @ W_1, gs._COO)
        att1 = torch.sum(res1, dim=1).unsqueeze(1)
        res2 = gs.ops.u_mul_v(subA, u_feats @ W_2, v_feats @ W_2, gs._COO)
        att2 = torch.sum(res2, dim=1).unsqueeze(1)
        att3 = subA._graph._CAPI_normalize(0, gs._CSC)._CAPI_get_data("default").unsqueeze(1)
        att = torch.cat([att1, att2, att3], dim=1)
        att = F.relu(att @ F.softmax(sample_a, dim=0))
        att = att + 10e-10 * torch.ones_like(att)
        subA._graph._CAPI_set_data(att)
        subg = subA._graph._CAPI_sampling_with_probs(0, att, fanout, True, gs._CSC, gs._CSC)
        unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subg._CAPI_relabel()
        seeds = unique_tensor
    input_nodes = seeds
    return input_nodes, output_nodes, blocks


def benchmark(args, graph, nid, fanouts, n_epoch, features, W1, W2, Wa, sampler):
    print(f"####################################################{sampler.__name__}")
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
            input_nodes, output_nodes, blocks = sampler(graph, seeds, fanouts, features, W1, W2, Wa, args.use_uva)

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
    if features == None:
        features = torch.rand(g.num_nodes(), 128, dtype=torch.float32)
    features = features.to(device)
    W1 = torch.nn.init.xavier_normal_(torch.Tensor(features.shape[1], 64)).to("cuda")
    W2 = torch.nn.init.xavier_normal_(torch.Tensor(features.shape[1], 64)).to("cuda")
    Wa = torch.FloatTensor([[10e-3], [10e-3], [10e-1]]).to("cuda")
    csc_indptr, csc_indices, edge_ids = g.adj_sparse("csc")
    if args.use_uva and device == "cpu":
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
        features = features.pin_memory()
    m = gs.Matrix(gs.Graph(False))
    m._graph._CAPI_load_csc(csc_indptr, csc_indices)
    print("Check load successfully:", m._graph._CAPI_metadata(), "\n")

    n_epoch = args.num_epoch
    if args.dataset == "livejournal" or args.dataset == "ogbn-products":
        benchmark(args, m, train_nid, fanouts, n_epoch, features, W1, W2, Wa, sample_w_o_relabel)
    else:
        benchmark(args, m, train_nid, fanouts, n_epoch, features, W1, W2, Wa, sample_w_relabel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=6, help="run how many epochs")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Training model on gpu or cpu")
    parser.add_argument("--use-uva", type=bool, default=False, help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset", default="ogbn-products", help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=64, help="batch size for training")
    parser.add_argument("--samples", default="10,10", help="sample size for each layer")
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
