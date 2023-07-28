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


def sage_batchsampler(A: gs.Graph, seeds, seeds_ptr, fanouts):
    ptrts, indts = [], []
    for layer, fanout in enumerate(fanouts):
        subg = A._graph._CAPI_fused_columnwise_slicing_sampling(seeds, fanout, False)
        indptr, indices, eids = subg._CAPI_get_csc()
        indices_ptr = indptr[seeds_ptr]

        ptrt = torch.ops.gs_ops.IndptrSplitByOffset(indptr, seeds_ptr)
        indt = torch.ops.gs_ops.SplitByOffset(indices, indices_ptr)
        ptrts.append(ptrt)
        indts.append(indt)
        seeds, seeds_ptr = indices, indices_ptr
    return ptrts, indts


def benchmark_w_o_relabel(args, matrix, nid):
    print('###################################################gSampler deepwalk')
    print(f"train id size: {len(nid)}, dataset:{args.dataset}")
    batch_size = args.big_batch
    seedloader = SeedGenerator(
        nid, batch_size=batch_size, shuffle=True, drop_last=False)
    fanouts = [int(x.strip()) for x in args.samples.split(',')]
    small_batch_size = args.batchsize
    num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)
    orig_seeds_ptr = torch.arange(num_batches + 1, dtype=torch.int64, device='cuda') * small_batch_size
    print(args.num_epoch, batch_size, small_batch_size, fanouts)
    epoch_time = []
    mem_list = []
    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print('memory allocated before training:',
          static_memory / (1024 * 1024 * 1024), 'GB')
    for epoch in range(args.num_epoch):
        torch.cuda.reset_peak_memory_stats()
        seeds = torch.arange(512).to('cuda')
        torch.cuda.synchronize()
        start = time.time()
        for it, seeds in enumerate(tqdm.tqdm(seedloader)):
        # for i in range(12816):
        #     seeds = seeds.to('cuda')
            seeds_ptr = orig_seeds_ptr
            if it == len(seedloader) - 1:
                num_batches = int((seeds.numel() + small_batch_size - 1) / small_batch_size)
                seeds_ptr = torch.arange(num_batches + 1,
                                         dtype=torch.int64,
                                         device='cuda') * small_batch_size
                seeds_ptr[-1] = seeds.numel()
            ptrts, indts = sage_batchsampler(matrix, seeds,seeds_ptr,fanouts)
            # print(len(ptrts[0][0]),len(indts[0][0]))
            # print(len(ptrts[1][0]),len(indts[1][0]))

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
        log_info = ["gSampler", args.dataset, np.mean(epoch_time[1:]), "sage"]
        writer.writerow(log_info)
        print(f"result writen to ../outputs/result.csv")
        
def load(dataset,args):
    device = args.device
    use_uva = args.use_uva
    g, features, labels, n_classes, splitted_idx = dataset
    sample_list = []
    static_memory = torch.cuda.memory_allocated()
    train_nid = splitted_idx['train']
    

    csc_indptr, csc_indices, edge_ids = g.adj_sparse('csc')
    if args.data_type == 'int':
        # csc_indptr = csc_indptr.int()
        csc_indices=csc_indices.int()
        train_nid = train_nid.int()
        # print("convect to csc")
        # g = g.formats("csc")
        # print("after convert to csc")
    else:
        csc_indptr = csc_indptr.long()
        csc_indices=csc_indices.long()
        train_nid = train_nid.long()
        # g = g.formats("csc")
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
    parser.add_argument("--batchsize", type=int, default=512,
                        help="batch size for training")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="numbers of workers for sampling, must be 0 when gpu or uva is used")
    parser.add_argument("--num-epoch", type=int, default=6,
                        help="numbers of epoch in training")
    parser.add_argument("--sample-mode", default='ad-hoc', choices=['ad-hoc', 'fine-grained','matrix-fused','matrix-nonfused'],
                        help="sample mode")
    parser.add_argument("--data-type", default='long', choices=['int', 'long'],
                        help="data type")
    parser.add_argument("--samples",
                        default='25,10',
                        help="sample size for each layer")
    parser.add_argument("--big-batch", type=int, default=5120,
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
    load(dataset,args)
