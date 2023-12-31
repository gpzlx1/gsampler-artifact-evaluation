import gs
from gs.utils import SeedGenerator, load_graph
import torch
import numpy as np
import time
import tqdm
import argparse
import csv
from typing import List


def ladies_sampler(A: gs.BatchMatrix, fanouts: List, seeds: torch.Tensor,
                   seeds_ptr: torch.Tensor):
    ret = []
    for K in fanouts:
        subA = A[:, seeds::seeds_ptr]
        subA.edata["p"] = subA.edata["w"]**2
        prob = subA.sum("p", axis=1)
        neighbors, probs_ptr = subA.all_rows()
        sampleA, select_index = subA.collective_sampling(
            K, prob, probs_ptr, False)
        sampleA = sampleA.div("w", prob[select_index], axis=1)
        out = sampleA.sum("w", axis=0)
        sampleA = sampleA.div("w", out, axis=0)
        seeds, seeds_ptr = sampleA.all_nodes()
        ret.append(sampleA)
    return ret


def benchmark(args, matrix, nid, fanouts, n_epoch, sampler):
    print("####################################################")
    batch_size = args.batching_batchsize
    small_batch_size = args.batchsize

    seedloader = SeedGenerator(nid,
                               batch_size=batch_size,
                               shuffle=True,
                               drop_last=False)
    num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)
    orig_seeds_ptr = torch.arange(
        num_batches + 1, dtype=torch.int64, device="cuda") * small_batch_size

    # compile
    compile_func = gs.jit.compile(func=sampler,
                                  args=(matrix, fanouts,
                                        seedloader.data[:batch_size],
                                        orig_seeds_ptr))

    epoch_time = []
    mem_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print("memory allocated before training:",
          static_memory / (1024 * 1024 * 1024), "GB")
    for epoch in range(n_epoch):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        for it, seeds in enumerate(tqdm.tqdm(seedloader)):
            seeds_ptr = orig_seeds_ptr
            if it == len(seedloader) - 1:
                num_batches = int(
                    (seeds.numel() + small_batch_size - 1) / small_batch_size)
                seeds_ptr = torch.arange(num_batches + 1,
                                         dtype=torch.int64,
                                         device="cuda") * small_batch_size
                seeds_ptr[-1] = seeds.numel()
            outputs = compile_func(matrix, fanouts, seeds, seeds_ptr)

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        mem_list.append((torch.cuda.max_memory_allocated() - static_memory) /
                        (1024 * 1024 * 1024))

        print(
            "Epoch {:05d} | Epoch Sample Time {:.4f} s | GPU Mem Peak {:.4f} GB"
            .format(epoch, epoch_time[-1], mem_list[-1]))

    with open("outputs/result.csv", "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        # system name, dataset, sampling time, mem peak
        log_info = [
            "gSampler", args.dataset,
            np.mean(epoch_time[1:]),
            np.mean(mem_list[1:])
        ]
        writer.writerow(log_info)

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
    m = gs.Matrix()
    m.load_graph("CSC", [csc_indptr, csc_indices])
    m.edata["w"] = weight
    bm = gs.BatchMatrix()
    bm.load_from_matrix(m)

    n_epoch = args.num_epoch
    benchmark(args, bm, train_nid, fanouts, n_epoch, ladies_sampler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch",
                        type=int,
                        default=6,
                        help="run how many epochs")
    parser.add_argument("--device",
                        default="cuda",
                        choices=["cuda", "cpu"],
                        help="Training model on gpu or cpu")
    parser.add_argument(
        "--use-uva",
        type=bool,
        default=False,
        help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset",
                        default="ogbn-products",
                        help="which dataset to load for training")
    parser.add_argument("--batchsize",
                        type=int,
                        default=512,
                        help="batch size for training")
    parser.add_argument("--batching-batchsize",
                        type=int,
                        default=12800,
                        help="batch size for training")
    parser.add_argument("--samples",
                        default="4000,4000,4000",
                        help="sample size for each layer")
    args = parser.parse_args()
    print(args)

    if args.dataset.startswith("ogbn"):
        dataset = load_graph.load_ogb(args.dataset, "/home/ubuntu/dataset")
    elif args.dataset == "livejournal":
        dataset = load_graph.load_dglgraph(
            "/home/ubuntu/dataset/livejournal/livejournal.bin")
    elif args.dataset == "friendster":
        dataset = load_graph.load_dglgraph(
            "/home/ubuntu/dataset/friendster/friendster.bin")
    else:
        raise NotImplementedError
    print(dataset[0])
    train(dataset, args)
