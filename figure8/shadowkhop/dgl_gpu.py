from gs.utils import SeedGenerator, load_graph
import torch
import torch.nn.functional as F
from dgl.utils import gather_pinned_tensor_rows
import numpy as np
import time
import tqdm
import argparse
import dgl
import csv


def benchmark(args, graph, nid, fanouts, n_epoch):
    print("####################################################START")
    sampler = dgl.dataloading.ShaDowKHopSampler(fanouts)
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
            input_nodes, output_nodes, blocks = sampler.sample(graph, seeds)

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        mem_list.append((torch.cuda.max_memory_allocated() - static_memory) / (1024 * 1024 * 1024))

        print("Epoch {:05d} | Epoch Sample Time {:.4f} s | GPU Mem Peak {:.4f} GB".format(epoch, epoch_time[-1], mem_list[-1]))

    tag = "CPU" if (args.device == "cpu" and not args.use_uva) else "DGL"
    with open("outputs/result.csv", "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        # system name, dataset, sampling time, mem peak
        log_info = [tag, args.dataset, np.mean(epoch_time[1:]), np.mean(mem_list[1:])]
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
    train_nid = splitted_idx["train"].to(device)
    csc_indptr, csc_indices, edge_ids = g.adj_sparse("csc")
    if args.use_uva and device == "cpu":
        g.pin_memory_()
        train_nid = train_nid.cuda()

    n_epoch = args.num_epoch
    benchmark(args, g, train_nid, fanouts, n_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=6, help="run how many epochs")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Training model on gpu or cpu")
    parser.add_argument("--use-uva", type=bool, default=False, help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset", default="ogbn-products", help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=512, help="batch size for training")
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
