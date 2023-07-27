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
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_ogbn_products():
    data = DglNodePropPredDataset(name="ogbn-products",root="/home/ubuntu/dataset")
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    g=g.long()
    feat = g.ndata['feat']
    labels = labels[:, 0]
    n_classes = len(
        torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    g.ndata.clear()
    # print("before:",g)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    # print("after:",g)
    # sp.save_npz("/home/ubuntu/data/products_adj.npz", g.adj(scipy_fmt='coo'))
    return g, feat, labels, n_classes, splitted_idx

def load_100Mpapers():
    train_id = torch.load("/home/ubuntu/dataset/papers100m_train_id.pt")
    splitted_idx = dict()
    splitted_idx['train']=train_id
    coo_matrix = sp.load_npz("/home/ubuntu/dataset/ogbn-papers100M_adj.npz")
    g = dgl.from_scipy(coo_matrix)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g=g.long()
    return g, None, None, None, splitted_idx
def load_livejournal():
    train_id = torch.load("/home/ubuntu/dataset/livejournal_trainid.pt")
    splitted_idx = dict()
    splitted_idx['train']=train_id
    coo_matrix = sp.load_npz("/home/ubuntu/dataset/livejournal/livejournal_adj.npz")

    g = dgl.from_scipy(coo_matrix)

    # g = g.formats("csc")
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    print("after:",g)
    sp.save_npz("/home/ubuntu/dataset/livejournal/livejournal_adj.npzcon", g.adj(scipy_fmt='coo'))
    g=g.long()
    return g, None, None, None, splitted_idx

def load_friendster():
    train_id = torch.load("/home/ubuntu/dataset/friendster_trainid.pt")
    splitted_idx = dict()
    splitted_idx['train']=train_id
    # bin_path = "/home/ubuntu/data/friendster/friendster.bin"
    # g_list, _ = dgl.load_graphs(bin_path)
    # g = g_list[0]
    # print("graph loaded")
    # train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    # test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
    # val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]

    # features = np.random.rand(g.num_nodes(), 128)
    # labels = np.random.randint(0, 3, size=g.num_nodes())
    # feat = torch.tensor(features, dtype=torch.float32)
    # labels = torch.tensor(labels, dtype=torch.int64)
    # n_classes = 3

    coo_matrix = sp.load_npz("/home/ubuntu/dataset/friendster/friendster_adj.npz")
    # csr_matrix = coo_matrix.tocsr()
    # sp.save_npz("/home/ubuntu/data/friendster/friendster_adj_csr.npz",csr_matrix)
    # print("file saved!")
    g = dgl.from_scipy(coo_matrix)
    # g = dgl.from_scipy(coo_matrix)
    print(g.formats())
    # g = g.formats("csc")
    g=g.long()
    return g, None,None,None,splitted_idx

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
        log_info = ["dgl", args.dataset, np.mean(epoch_time[1:]), "rw"]
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
