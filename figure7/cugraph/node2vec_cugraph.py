import cudf
import numpy as np
import cugraph
from dgl.sampling import node2vec_random_walk
import time
import dgl
import torch
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.sparse as sp


def load_ogb(name):
    data = DglNodePropPredDataset(name=name)
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    feat = g.ndata["feat"]
    labels = labels[:, 0]
    n_classes = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    g.ndata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, feat, labels, n_classes, splitted_idx


def load_livejournal():
    train_id = torch.load("/home/ubuntu/.dgl/livejournal_trainid.pt")
    splitted_idx = dict()
    splitted_idx["train"] = train_id
    coo_matrix = sp.load_npz("/home/ubuntu/.dgl/livejournal_adj.npz")
    g = dgl.from_scipy(coo_matrix)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.long()
    return g, None, None, None, splitted_idx


def time_randomwalk(graph, seeds, batchsize, walk_length, batchnum):
    """
    Test cost time of random walk
    """
    runs = 6
    time_list = []
    for i in range(runs):
        torch.cuda.synchronize()
        start_time = time.time()
        for j in range(batchnum):
            start = j * batchsize
            end = seeds.shape[0] if j == batchnum - 1 else (j + 1) * batchsize
            sub_slicing = seeds[start:end]
            paths, weights, path_sizes = cugraph.node2vec(
                graph,
                start_vertices=sub_slicing,
                max_depth=walk_length,
                compress_result=True,
                p=2.0,
                q=0.5,
            )
        torch.cuda.synchronize()
        end_time = time.time()
        time_list.append(end_time - start_time)
        print(
            "Run {} seeds, {} times, epoch run time: {:.6f} ms".format(
                len(seeds), batchnum, time_list[-1] * 1000
            )
        )
    print("average epoch run time:", np.mean(time_list[1:]) * 1000)


# dataset = load_ogb('ogbn-products')
dataset = load_livejournal()
dgl_graph = dataset[0]
train_id = dataset[4]["train"]
train_id = train_id.cpu().numpy()
index = np.random.permutation(train_id.shape[0])
train_id = cudf.Series(train_id[index])
dgl_graph = dgl_graph.to("cuda")
g_cugraph = dgl_graph.to_cugraph()
del dgl_graph
print("Timing random walks")
batchsize = 128
walk_len = 80
batchnum = int((train_id.shape[0] + batchsize - 1) / batchsize)
time_randomwalk(g_cugraph, train_id, batchsize, 80, batchnum)
# coo_matrix = sp.load_npz("/home/ubuntu/data/friendster/friendster_adj.npz")
# g = dgl.from_scipy(coo_matrix)
# g = g.formats("csc")
