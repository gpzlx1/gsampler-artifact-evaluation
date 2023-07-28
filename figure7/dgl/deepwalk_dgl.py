import torch
import dgl
from gs.utils import load_graph
import time
import numpy as np
import gs
from gs.utils import SeedGenerator
import sys 
sys.path.append("..") 
from ogb.nodeproppred import DglNodePropPredDataset
import argparse
from dgl.dataloading import DataLoader, NeighborSampler
import tqdm
import scipy.sparse as sp
import csv
from load_graph_utils import load_ogbn_products,load_livejournal,load_100Mpapers,load_friendster

class DeepWalkSampler(object):
    def __init__(self, walk_length=80):
        super().__init__()
        self.sampling_time = 0
        self.walk_length = walk_length
    def sample(self, g, seeds,exclude_eids=None):
        # print("begin sample")
        # print("shape:",seeds.shape)
        traces, types = dgl.sampling.random_walk(g, nodes=seeds, length=self.walk_length)
        # print("after sample")
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('dgl transform and unique')
        # sampled_nodes = traces.view(traces.numel())
        # sampled_nodes = sampled_nodes[sampled_nodes !=-1]
        # sampled_nodes = torch.unique(sampled_nodes, sorted=False)
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('dgl subgraph')
        # sg = g.subgraph(sampled_nodes, relabel_nodes=True)
        # torch.cuda.nvtx.range_pop()
        return traces



def benchmark_w_o_relabel(args, graph, nid):
    print('####################################################DGL deepwalk')
    sampler = DeepWalkSampler(args.walk_length)
    print("train id size:",len(nid))
    # print(nid.shape)
    # seedloader = SeedGenerator(
    #     nid, batch_size=args.batchsize, shuffle=True, drop_last=True)
    seedloader = DataLoader(graph, nid, sampler,batch_size=args.batchsize, use_prefetch_thread=False,
    shuffle=True,drop_last=False,device='cuda',use_uva=args.use_uva)
    # train_dataloader = DataLoader(g, train_nid, sampler,batch_size=config['batch_size'], use_prefetch_thread=False,
    # shuffle=False,drop_last=False, num_workers=config['num_workers'],device='cuda',use_uva=config['use_uva'])
    epoch_time = []
    mem_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')
    for epoch in range(args.num_epoch):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        for it, (traces) in enumerate(tqdm.tqdm(seedloader)):
            pass
            # print("iteration:",it)
            # traces = sampler.sample(graph, seeds)

        torch.cuda.synchronize()
        epoch_time.append(time.time() - start)
        mem_list.append((torch.cuda.max_memory_allocated() -
                        static_memory) / (1024 * 1024 * 1024))

        print("Epoch {:05d} | Epoch Sample Time {:.4f} s | GPU Mem Peak {:.4f} GB"
              .format(epoch, epoch_time[-1], mem_list[-1]))

    # use the first epoch to warm up
    print('Average epoch sampling time:', np.mean(epoch_time[1:])*1000," ms")
    print('Average epoch gpu mem peak:', np.mean(mem_list[1:])," GB")
    print('####################################################END')
    
    with open("../outputs/result.csv", "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        # system name, dataset, sampling time, mem peak
        log_info = ["DGL", args.dataset, np.mean(epoch_time[1:]), "rw"]
        writer.writerow(log_info)
        print(f"result writen to ../outputs/result.csv")

    

def load(dataset,args):
    device = args.device
    use_uva = args.use_uva
    g, features, labels, n_classes, splitted_idx = dataset
    sample_list = []
    static_memory = torch.cuda.memory_allocated()
    train_nid = splitted_idx['train']
    if args.data_type == 'int':
        g = g.int()
        train_nid = train_nid.int()
        # print("convect to csc")
        # g = g.formats("csc")
        # print("after convert to csc")
    else:
        g = g.long()
        train_nid = train_nid.long()
        # g = g.formats("csc")

    g = g.to(device)
    train_nid = train_nid.to('cuda')

    # csc_indptr, csc_indices, edge_ids = g.adj_sparse('csc')
    if use_uva and device == 'cpu':
        g=g.pin_memory_()
         # csc_indptr = csc_indptr.pin_memory()
        # csc_indices = csc_indices.pin_memory()
    benchmark_w_o_relabel(args, g, train_nid)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument('--use-uva', action=argparse.BooleanOptionalAction,
                        help="Wether to use UVA to sample graph and load feature")
    parser.add_argument("--dataset", default='products', choices=['reddit', 'products', 'papers100m','friendster','livejournal'],
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=1024,
                        help="batch size for training")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="numbers of workers for sampling, must be 0 when gpu or uva is used")
    parser.add_argument("--num-epoch", type=int, default=3,
                        help="numbers of epoch in training")
    parser.add_argument("--sample-mode", default='ad-hoc', choices=['ad-hoc', 'fine-grained','matrix-fused','matrix-nonfused'],
                        help="sample mode")
    parser.add_argument("--data-type", default='long', choices=['int', 'long'],
                        help="data type")
    parser.add_argument("--walk-length", type=int, default=80,
                        help="random walk walk length")
    args = parser.parse_args()
    print('Loading data')
    if args.dataset == 'products':
        dataset = load_ogbn_products()
    elif args.dataset == 'papers100m':
        dataset = load_100Mpapers()
    elif args.dataset == 'friendster':
        dataset = load_friendster()
    elif args.dataset == 'livejournal':
        dataset = load_livejournal()
    print(dataset[0])


# bench('DGL random walk', dgl_sampler, g, 4, iters=10, node_idx=nodes)
# bench('Matrix random walk Non-fused', matrix_sampler_nonfused, matrix,
#       4, iters=10, node_idx=nodes)
    load(dataset,args)
