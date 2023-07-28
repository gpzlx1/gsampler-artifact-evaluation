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


class node2vecSampler(object):
    def __init__(self,walk_length, p=0.5,q=2,):
        super().__init__()
        self.walk_length = walk_length
        self.p=p
        self.q=q
    def sample(self, g, seeds,exclude_eids=None):
        torch.cuda.nvtx.range_push('dgl random walk')
        traces = dgl.sampling.node2vec_random_walk(g, seeds, self.p,self.q,self.walk_length)
        return traces




def benchmark_w_o_relabel(args, graph, nid):
    print('####################################################DGL node2vec')
    sampler = node2vecSampler(args.walk_length)
    print("train id size:",len(nid))
    # seedloader = SeedGenerator(
    #     nid, batch_size=args.batchsize, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(graph, nid, sampler,batch_size=args.batchsize, use_prefetch_thread=False,
    shuffle=True,drop_last=False, num_workers=args.num_workers,device='cpu',use_uva=False)
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
        # with train_dataloader.enable_cpu_affinity():
        for it, seeds in enumerate(tqdm.tqdm(train_dataloader)):
            pass
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
        log_info = ["DGL", args.dataset, np.mean(epoch_time[1:]), "node2vec"]
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
    else:
        g = g.long()
        train_nid = train_nid.long()
    # g = g.to(device)
    # train_nid = train_nid.to('cuda')
    # csc_indptr, csc_indices, edge_ids = g.adj_sparse('csc')
    if use_uva and device == 'cpu':
        g.pin_memory_()
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
    parser.add_argument("--batchsize", type=int, default=128,
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
