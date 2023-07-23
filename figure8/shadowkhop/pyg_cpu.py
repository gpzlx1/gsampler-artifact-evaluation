import torch
import numpy as np
from gs.utils import load_graph
from utils import from_dgl
from torch_geometric.loader import ShaDowKHopSampler
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm
import argparse
import time


def train(args, dataset):
    # kwargs = {"batch_size": args.batchsize, "num_workers": 2, 'persistent_workers': True}
    kwargs = {"batch_size": args.batchsize}

    graph, train_idx = dataset
    # graph = graph.to('cuda')

    train_loader = ShaDowKHopSampler(graph, depth=2, num_neighbors=10, node_idx=train_idx, replace=False, **kwargs)
    print("####################################################START")
    epoch_time = []
    mem_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print("memory allocated before training:", static_memory / (1024 * 1024 * 1024), "GB")
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epoch):
        torch.cuda.synchronize()
        start = time.time()
        for it, data in enumerate(tqdm(train_loader)):
            pass
        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        mem_list.append((torch.cuda.max_memory_allocated() - static_memory) / (1024 * 1024 * 1024))

        print("Epoch {:05d} | Epoch Sample Time {:.4f} s | GPU Mem Peak {:.4f} GB".format(epoch, epoch_time[-1], mem_list[-1]))

    # use the first epoch to warm up
    print("Average epoch sampling time:", np.mean(epoch_time[1:]))
    print("Average epoch gpu mem peak:", np.mean(mem_list[1:]))
    print("####################################################END")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ogbn-products", help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=512, help="batch size for training")
    parser.add_argument("--epoch", type=int, default=6, help="number of epoch")
    args = parser.parse_args()
    print(args)

    if args.dataset.startswith("ogbn"):
        dataset = PygNodePropPredDataset(args.dataset, "/home/ubuntu/dataset")
        split_idx = dataset.get_idx_split()
        dataset = (dataset[0], split_idx["train"])
    elif args.dataset == "livejournal":
        dataset = load_graph.load_dglgraph("/home/ubuntu/dataset/livejournal/livejournal.bin")
        g, features, labels, n_classes, splitted_idx = dataset
        dataset = (from_dgl(g), splitted_idx["train"])
    elif args.dataset == "friendster":
        dataset = load_graph.load_dglgraph("/home/ubuntu/dataset/friendster/friendster.bin")
        g, features, labels, n_classes, splitted_idx = dataset
        dataset = (from_dgl(g), splitted_idx["train"])
    else:
        raise NotImplementedError
    print(dataset[0])
    train(args, dataset)
