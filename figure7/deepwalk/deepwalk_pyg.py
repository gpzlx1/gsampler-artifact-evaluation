import torch
import argparse
from gs.utils import load_graph, SeedGenerator
import time
from tqdm import tqdm
import pyg_lib
from pyg_lib import sampler
import numpy as np
import csv

def benchmark(args, graph, nid, fanouts, n_epoch):
    print("####################################################START")
    indptr, indices = graph
    seedloader = SeedGenerator(
        nid, batch_size=args.batchsize, shuffle=True, drop_last=False
    )

    epoch_time = []
    mem_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print(
        "memory allocated before training:", static_memory / (1024 * 1024 * 1024), "GB"
    )
    for epoch in range(n_epoch):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        for it, seeds in enumerate(tqdm(seedloader)):
            res = sampler.random_walk(indptr, indices, seeds, 80)

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        mem_list.append(
            (torch.cuda.max_memory_allocated() - static_memory) / (1024 * 1024 * 1024)
        )

        print(
            "Epoch {:05d} | Epoch Sample Time {:.4f} s | GPU Mem Peak {:.4f} GB".format(
                epoch, epoch_time[-1], mem_list[-1]
            )
        )

    # use the first epoch to warm up
    print("Average epoch sampling time:", np.mean(epoch_time[1:]))
    print("Average epoch gpu mem peak:", np.mean(mem_list[1:]))
    print("####################################################END")
    with open("../outputs/result.csv", "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        # system name, dataset, sampling time, mem peak
        log_info = ["pyg", args.dataset, np.mean(epoch_time[1:]), "rw"]
        writer.writerow(log_info)
        print(f"result writen to ../outputs/result.csv")


def train(dataset, args):
    device = args.device
    use_uva = args.use_uva
    fanouts = [int(x.strip()) for x in args.samples.split(",")]

    g, features, labels, n_classes, splitted_idx = dataset
    g = g.long().to(device)
    train_nid = splitted_idx["train"].to('cuda')
    csc_indptr, csc_indices, edge_ids = g.adj_sparse("csc")
    if use_uva and device == "cpu":
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
    graph = (csc_indptr, csc_indices)

    n_epoch = 6
    benchmark(args, graph, train_nid, fanouts, n_epoch)


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
        "--dataset", default="ogbn-products", help="which dataset to load for training"
    )
    parser.add_argument(
        "--batchsize", type=int, default=1024, help="batch size for training"
    )
    parser.add_argument("--samples", default="1", help="sample size for each layer")
    args = parser.parse_args()
    print(args)

    if args.dataset.startswith("ogbn"):
        dataset = load_graph.load_ogb(args.dataset, "/home/ubuntu/dataset")
    elif args.dataset == "livejournal":
        dataset = load_graph.load_dglgraph(
            "/home/ubuntu/dataset/livejournal/livejournal.bin"
        )
    elif args.dataset == "friendster":
        dataset = load_graph.load_dglgraph(
            "/home/ubuntu/dataset/friendster/friendster.bin"
        )
    else:
        raise NotImplementedError
    print(dataset[0])
    train(dataset, args)
