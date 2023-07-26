import gs
from gs.utils import SeedGenerator, load_graph
import torch
import torch.nn.functional as F
from dgl.utils import gather_pinned_tensor_rows
import numpy as np
import time
import tqdm
import argparse
import dgl
import numba
from numba.core import types
from numba.typed import Dict
import csv


@numba.njit
def find_indices_in(a, b):
    d = Dict.empty(key_type=types.int64, value_type=types.int64)
    for i, v in enumerate(b):
        d[v] = i
    ai = np.zeros_like(a)
    for i, v in enumerate(a):
        ai[i] = d.get(v, -1)
    return ai


class PASSSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts, W_1, W_2, sample_a, use_uva, features=None):
        super().__init__()
        self.fanouts = fanouts
        self.W_1 = W_1
        self.W_2 = W_2
        self.sample_a = sample_a
        self.use_uva = use_uva
        self.features = features
        self.ret_loss_tuple = None

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            subg = dgl.in_subgraph(g, seed_nodes)
            edges = subg.edges()
            nodes = torch.unique(edges[0])

            if self.use_uva:
                u_feats = gather_pinned_tensor_rows(self.features, nodes)
                v_feats = gather_pinned_tensor_rows(self.features, seed_nodes)
            else:
                u_feats = self.features[nodes]
                v_feats = self.features[seed_nodes]
            u_feats_all_w1 = torch.empty((subg.num_nodes(), self.W_1.shape[1]), dtype=torch.float32, device=u_feats.device)
            v_feats_all_w1 = torch.empty((subg.num_nodes(), self.W_1.shape[1]), dtype=torch.float32, device=v_feats.device)
            u_feats_all_w1[nodes] = u_feats @ self.W_1
            v_feats_all_w1[seed_nodes] = v_feats @ self.W_1
            u_feats_all_w2 = torch.empty((subg.num_nodes(), self.W_2.shape[1]), dtype=torch.float32, device=u_feats.device)
            v_feats_all_w2 = torch.empty((subg.num_nodes(), self.W_2.shape[1]), dtype=torch.float32, device=v_feats.device)
            u_feats_all_w2[nodes] = u_feats @ self.W_2
            v_feats_all_w2[seed_nodes] = v_feats @ self.W_2

            att1 = torch.sum(dgl.ops.u_mul_v(subg, u_feats_all_w1, v_feats_all_w1), dim=1).unsqueeze(1)
            att2 = torch.sum(dgl.ops.u_mul_v(subg, u_feats_all_w2, v_feats_all_w2), dim=1).unsqueeze(1)
            subg.ndata["v"] = subg.in_degrees()
            subg.apply_edges(lambda edges: {"w": 1 / edges.dst["v"]})
            att3 = subg.edata["w"].unsqueeze(1)
            att = torch.cat([att1, att2, att3], dim=1)
            att = F.relu(att @ F.softmax(self.sample_a, dim=0))
            att = att + 10e-10 * torch.ones_like(att)

            frontier = dgl.sampling.sample_neighbors(subg, seed_nodes, fanout, prob=att, replace=True)
            block = dgl.to_block(frontier, seed_nodes)
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        input_nodes = seed_nodes
        return input_nodes, output_nodes, blocks


class PASSSamplerRelabel(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts, W_1, W_2, sample_a, use_uva, features=None):
        super().__init__()
        self.fanouts = fanouts
        self.W_1 = W_1
        self.W_2 = W_2
        self.sample_a = sample_a
        self.use_uva = use_uva
        self.features = features
        self.ret_loss_tuple = None

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            subg = dgl.in_subgraph(g, seed_nodes, relabel_nodes=True)
            edges = subg.edges()
            nodes = torch.unique(edges[0])
            global_nid = subg.ndata[dgl.NID][nodes]

            if seed_nodes.device == torch.device("cuda:0"):
                local_seeds_nid = torch.ops.gs_ops.index_search(subg.ndata[dgl.NID], seed_nodes)
            else:
                local_seeds_nid = find_indices_in(seed_nodes.numpy(), subg.ndata[dgl.NID].numpy())
                local_seeds_nid = torch.from_numpy(local_seeds_nid)

            if self.use_uva:
                u_feats = gather_pinned_tensor_rows(self.features, global_nid)
                v_feats = gather_pinned_tensor_rows(self.features, seed_nodes)
            else:
                u_feats = self.features[global_nid]
                v_feats = self.features[seed_nodes]
            u_feats_all_w1 = torch.empty((subg.num_nodes(), self.W_1.shape[1]), dtype=torch.float32, device=u_feats.device)
            v_feats_all_w1 = torch.empty((subg.num_nodes(), self.W_1.shape[1]), dtype=torch.float32, device=v_feats.device)
            u_feats_all_w1[nodes] = u_feats @ self.W_1
            v_feats_all_w1[local_seeds_nid] = v_feats @ self.W_1
            u_feats_all_w2 = torch.empty((subg.num_nodes(), self.W_2.shape[1]), dtype=torch.float32, device=u_feats.device)
            v_feats_all_w2 = torch.empty((subg.num_nodes(), self.W_2.shape[1]), dtype=torch.float32, device=v_feats.device)
            u_feats_all_w2[nodes] = u_feats @ self.W_2
            v_feats_all_w2[local_seeds_nid] = v_feats @ self.W_2

            att1 = torch.sum(dgl.ops.u_mul_v(subg, u_feats_all_w1, v_feats_all_w1), dim=1).unsqueeze(1)
            att2 = torch.sum(dgl.ops.u_mul_v(subg, u_feats_all_w2, v_feats_all_w2), dim=1).unsqueeze(1)
            subg.ndata["v"] = subg.in_degrees()
            subg.apply_edges(lambda edges: {"w": 1 / edges.dst["v"]})
            att3 = subg.edata["w"].unsqueeze(1)
            att = torch.cat([att1, att2, att3], dim=1)
            att = F.relu(att @ F.softmax(self.sample_a, dim=0))
            att = att + 10e-10 * torch.ones_like(att)

            frontier = dgl.sampling.sample_neighbors(subg, local_seeds_nid, fanout, prob=att, replace=True)
            block = dgl.to_block(frontier, local_seeds_nid)
            seed_nodes = frontier.ndata[dgl.NID][block.srcdata[dgl.NID]]
            blocks.insert(0, block)
        input_nodes = seed_nodes
        return input_nodes, output_nodes, blocks


def benchmark(args, graph, nid, fanouts, n_epoch, features, W1, W2, Wa, sampler_class):
    print(f"####################################################DGL {sampler_class.__name__}")
    sampler = sampler_class(fanouts, W_1=W1, W_2=W2, sample_a=Wa, use_uva=args.use_uva, features=features)
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
            input_nodes, output_nodes, blocks = sampler.sample_blocks(graph, seeds)

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
    if features == None:
        features = torch.rand(g.num_nodes(), 128, dtype=torch.float32)
    features = features.to(device)
    W1 = torch.nn.init.xavier_normal_(torch.Tensor(features.shape[1], 64)).to(device)
    W2 = torch.nn.init.xavier_normal_(torch.Tensor(features.shape[1], 64)).to(device)
    Wa = torch.FloatTensor([[10e-3], [10e-3], [10e-1]]).to(device)
    csc_indptr, csc_indices, edge_ids = g.adj_sparse("csc")
    if args.use_uva and device == "cpu":
        g.pin_memory_()
        features = features.pin_memory()
        train_nid = train_nid.cuda()
        W1, W2, Wa = W1.cuda(), W2.cuda(), Wa.cuda()

    n_epoch = args.num_epoch
    if args.dataset == "livejournal" or args.dataset == "ogbn-products":
        benchmark(args, g, train_nid, fanouts, n_epoch, features, W1, W2, Wa, PASSSampler)
    else:
        benchmark(args, g, train_nid, fanouts, n_epoch, features, W1, W2, Wa, PASSSamplerRelabel)


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
