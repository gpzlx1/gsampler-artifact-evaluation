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
from load_graph_utils import load_ogbn_products,load_livejournal,load_100Mpapers

def load_friendster():
    train_id = torch.load("/home/ubuntu/dataset/friendster_trainid_10x.pt")
    splitted_idx = dict()
    splitted_idx['train']=train_id
    bin_path = "/home/ubuntu/dataset/friendster/friendster_adj.bin"
    g_list, _ = dgl.load_graphs(bin_path)
    g = g_list[0]
    print("graph loaded")
    print(g.formats())
    # g = g.formats("csc")
    g=g.long()
    return g, None,None,None,splitted_idx

def matrix_batch_sampler_deepwalk(A: gs.Matrix, seeds, num_steps):
    path = A._graph._CAPI_node2vec_random_walk(seeds,num_steps,2,0.5)
    return path



def benchmark_w_o_relabel(args, matrix, nid):
    print('####################################################DGL deepwalk')
    # sampler = DeepWalkSampler(args.walk_length)
    print("train id size:",len(nid))
    batch_size = args.big_batch
    seedloader = SeedGenerator(
        nid, batch_size=batch_size, shuffle=True, drop_last=False)
    # train_dataloader = DataLoader(g, train_nid, sampler,batch_size=config['batch_size'], use_prefetch_thread=False,
    # shuffle=False,drop_last=False, num_workers=config['num_workers'],device='cuda',use_uva=config['use_uva'])
    

    small_batch_size = args.batchsize

    #orig_seeds_ptr = torch.arange(num_batches + 1, dtype=torch.int64, device='cuda') * small_batch_size
    print(args.num_epoch, batch_size, small_batch_size)
    
    epoch_time = []
    mem_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')
    for epoch in range(args.num_epoch):
        num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        for it, seeds in enumerate(tqdm.tqdm(seedloader)):
            seeds = seeds.to('cuda')
            if it == len(seedloader) - 1:
                break
                num_batches = int((seeds.numel() + small_batch_size - 1) / small_batch_size)
            paths = matrix_batch_sampler_deepwalk(matrix, seeds, args.walk_length)
            split_paths = torch.tensor_split(paths,num_batches)

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
        avg_sampling_time = np.mean(epoch_time[1:])
        if args.dataset == 'friendster':
            # to align with papers result
            avg_sampling_time /=10
        log_info = ["gSampler", args.dataset, avg_sampling_time, "node2vec"]
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
    csc_indptr, csc_indices, edge_ids = g.adj_sparse('csc')
    # train_nid = train_nid.int()
    # csc_indptr = csc_indptr.int()
    # csc_indices = csc_indices.int()
    if use_uva and device == 'cpu':
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
    else:
        csc_indptr = csc_indptr.to('cuda')
        csc_indices = csc_indices.to('cuda')

    m = gs.Matrix(gs.Graph(False))
    m._graph._CAPI_load_csc(csc_indptr, csc_indices)
    print("Check load successfully:", m._graph._CAPI_metadata(), '\n')
    train_nid = train_nid.to('cuda')
    # csc_indptr, csc_indices, edge_ids = g.adj_sparse('csc')
    del g
    benchmark_w_o_relabel(args, m, train_nid)



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
    parser.add_argument("--data-type", default='long', choices=['int', 'long'],
                        help="data type")
    parser.add_argument("--walk-length", type=int, default=80,
                        help="random walk walk length")
    parser.add_argument("--samples",
                        default='25,10',
                        help="sample size for each layer")
    parser.add_argument("--big-batch", type=int, default=1280,
                        help="big batch")
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
