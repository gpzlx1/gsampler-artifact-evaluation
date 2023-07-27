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


class ASGCNSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts, replace=False, use_uva=False, W=None, eweight=None, node_feats=None):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.use_uva = use_uva
        self.W = W
        self.edge_weight = eweight
        self.node_feats = node_feats

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        blocks = []
        output_nodes = seed_nodes
        features = self.node_feats
        for fanout in self.fanouts:
            subg = dgl.in_subgraph(g, seed_nodes)
            reversed_subg = dgl.reverse(subg, copy_edata=True)
            if self.use_uva:
                sampled_e_weight = gather_pinned_tensor_rows(self.edge_weight, reversed_subg.edata[dgl.EID])
            else:
                sampled_e_weight = self.edge_weight[reversed_subg.edata[dgl.EID]]
            p = torch.sqrt(dgl.ops.copy_e_sum(reversed_subg, sampled_e_weight**2))

            edges = subg.edges()
            nodes = torch.unique(edges[0])
            num_pick = np.min([nodes.numel(), fanout])
            if self.use_uva:
                node_feats_u = gather_pinned_tensor_rows(features, nodes)
                node_feats_v = gather_pinned_tensor_rows(features, seed_nodes)
            else:
                node_feats_u = features[nodes]
                node_feats_v = features[seed_nodes]
            h_u = node_feats_u @ self.W[:, 0]
            h_v = node_feats_v @ self.W[:, 1]
            h_v_sum = torch.sum(h_v)
            attention = torch.flatten((F.relu(h_u + h_v_sum) + 1) / fanout)
            g_u = torch.flatten(F.relu(h_u) + 1)

            q = F.normalize(p[nodes] * attention * g_u, p=1.0, dim=0)

            idx = torch.multinomial(q, num_pick, replacement=False)
            selected = nodes[idx]
            subg = dgl.out_subgraph(subg, selected)

            q_allnodes = torch.empty(subg.num_nodes(), dtype=torch.float32, device=subg.device)
            h_u_allnodes = torch.empty(subg.num_nodes(), dtype=torch.float32, device=subg.device)
            h_v_allnodes = torch.empty(subg.num_nodes(), dtype=torch.float32, device=subg.device)
            q_allnodes[selected] = q[idx]
            h_u_allnodes[selected] = h_u[idx]
            h_v_allnodes[seed_nodes] = h_v

            W_tilde = dgl.ops.u_add_v(subg, h_u_allnodes, h_v_allnodes)
            W_tilde = (F.relu(W_tilde) + 1) / num_pick
            W_tilde = dgl.ops.e_div_u(subg, W_tilde, q_allnodes)
            W_tilde = W_tilde * sampled_e_weight[subg.edata[dgl.EID]]
            # # reversed copy_e_sum
            # reversed_subg = dgl.reverse(subg, copy_edata=True)
            # u_sum = dgl.ops.copy_e_sum(reversed_subg, W_tilde)

            block = dgl.to_block(subg, seed_nodes)
            block.edata["w"] = W_tilde[block.edata[dgl.EID]]
            # block.srcdata['u_sum'] = u_sum[block.srcdata[dgl.NID]]
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        input_nodes = seed_nodes
        return input_nodes, output_nodes, blocks


class ASGCNSamplerRelabel(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts, replace=False, use_uva=False, W=None, eweight=None, node_feats=None):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.use_uva = use_uva
        self.W = W
        self.edge_weight = eweight
        self.node_feats = node_feats

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        blocks = []
        output_nodes = seed_nodes
        features = self.node_feats
        for fanout in self.fanouts:
            subg = dgl.in_subgraph(g, seed_nodes, relabel_nodes=True)
            reversed_subg = dgl.reverse(subg, copy_edata=True)
            if self.use_uva:
                sampled_e_weight = gather_pinned_tensor_rows(self.edge_weight, reversed_subg.edata[dgl.EID])
            else:
                sampled_e_weight = self.edge_weight[reversed_subg.edata[dgl.EID]]
            p = torch.sqrt(dgl.ops.copy_e_sum(reversed_subg, sampled_e_weight**2))

            edges = subg.edges()
            nodes = torch.unique(edges[0])
            num_pick = np.min([nodes.numel(), fanout])
            if self.use_uva:
                node_feats_u = gather_pinned_tensor_rows(features, subg.ndata[dgl.NID][nodes])
                node_feats_v = gather_pinned_tensor_rows(features, seed_nodes)
            else:
                node_feats_u = features[subg.ndata[dgl.NID][nodes]]
                node_feats_v = features[seed_nodes]
            h_u = node_feats_u @ self.W[:, 0]
            h_v = node_feats_v @ self.W[:, 1]
            h_v_sum = torch.sum(h_v)
            attention = torch.flatten((F.relu(h_u + h_v_sum) + 1) / fanout)
            g_u = torch.flatten(F.relu(h_u) + 1)

            q = F.normalize(p[nodes] * attention * g_u, p=1.0, dim=0)

            if seed_nodes.device == torch.device("cuda:0"):
                relabel_seeds_nodes = torch.ops.gs_ops.index_search(subg.ndata[dgl.NID], seed_nodes)
            else:
                relabel_seeds_nodes = find_indices_in(seed_nodes.numpy(), subg.ndata[dgl.NID].numpy())
                relabel_seeds_nodes = torch.from_numpy(relabel_seeds_nodes)

            idx = torch.multinomial(q, num_pick, replacement=False)
            selected = nodes[idx]
            subg = dgl.out_subgraph(subg, selected)

            q_allnodes = torch.empty(subg.num_nodes(), dtype=torch.float32, device=subg.device)
            h_u_allnodes = torch.empty(subg.num_nodes(), dtype=torch.float32, device=subg.device)
            h_v_allnodes = torch.empty(subg.num_nodes(), dtype=torch.float32, device=subg.device)
            q_allnodes[selected] = q[idx]
            h_u_allnodes[selected] = h_u[idx]
            h_v_allnodes[relabel_seeds_nodes] = h_v

            W_tilde = dgl.ops.u_add_v(subg, h_u_allnodes, h_v_allnodes)
            W_tilde = (F.relu(W_tilde) + 1) / num_pick
            W_tilde = dgl.ops.e_div_u(subg, W_tilde, q_allnodes)
            W_tilde = W_tilde * sampled_e_weight[subg.edata[dgl.EID]]
            # # reversed copy_e_sum
            # reversed_subg = dgl.reverse(subg, copy_edata=True)
            # u_sum = dgl.ops.copy_e_sum(reversed_subg, W_tilde)

            block = dgl.to_block(subg, relabel_seeds_nodes)
            block.edata["w"] = W_tilde[block.edata[dgl.EID]]
            # block.srcdata['u_sum'] = [block.srcdata[dgl.NID]]
            seed_nodes = subg.ndata[dgl.NID][block.srcdata[dgl.NID]]
            blocks.insert(0, block)
        input_nodes = seed_nodes
        return input_nodes, output_nodes, blocks


def benchmark(args, graph, nid, fanouts, n_epoch, adj_weight, features, W, sampler_class):
    print(f"####################################################DGL {sampler_class.__name__}")
    sampler = sampler_class(fanouts, replace=False, use_uva=args.use_uva, eweight=adj_weight, node_feats=features, W=W)
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
    # weight = normalized_laplacian_edata(g)
    weight = torch.ones(g.num_edges(), dtype=torch.float32, device=g.device)
    if features == None:
        features = torch.rand(g.num_nodes(), 128, dtype=torch.float32)
    features = features.to(device)
    W = torch.nn.init.xavier_normal_(torch.Tensor(features.shape[1], 2)).to(device)
    csc_indptr, csc_indices, edge_ids = g.adj_sparse("csc")
    weight = weight[edge_ids]
    if args.use_uva and device == "cpu":
        g.pin_memory_()
        weight, features = weight.pin_memory(), features.pin_memory()
        train_nid = train_nid.to("cuda")
        W = W.to("cuda")

    n_epoch = args.num_epoch
    if args.dataset != "ogbn-papers100M":
        benchmark(args, g, train_nid, fanouts, n_epoch, weight, features, W, ASGCNSampler)
    else:
        benchmark(args, g, train_nid, fanouts, n_epoch, weight, features, W, ASGCNSamplerRelabel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=6, help="run how many epochs")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Training model on gpu or cpu")
    parser.add_argument("--use-uva", type=bool, default=False, help="Wether to use UVA to sample graph and load feature")
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
