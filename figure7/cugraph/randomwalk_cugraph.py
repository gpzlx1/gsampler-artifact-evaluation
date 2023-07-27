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
<<<<<<< HEAD
# from cugraph.sampling.random_walks import uniform_random_walks

=======
import csv 
import argparse
>>>>>>> 88032ccd267c2bc12b1636703623dd3b7fbedd47

def load_ogb(name):
    data = DglNodePropPredDataset(name=name,root="/home/ubuntu/dataset/")
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
    train_id = torch.load("/home/ubuntu/dataset/livejournal_trainid.pt")
    splitted_idx = dict()
    splitted_idx["train"] = train_id
    coo_matrix = sp.load_npz("/home/ubuntu/dataset/livejournal_adj.npz")
    g = dgl.from_scipy(coo_matrix)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.long()
    return g, None, None, None, splitted_idx


def time_randomwalk(graph, seeds, batchsize, walk_length, batchnum):
    """
    Test cost time of random walk
    """
    runs = 2
    time_list = []
    sample_list = []
    for i in range(runs):
        torch.cuda.synchronize()
        start_time = time.time()
        epoch_sample_time=0
        for j in range(batchnum):
            start = j * batchsize
            end = seeds.shape[0] if j == batchnum - 1 else (j + 1) * batchsize
            sub_slicing = seeds[start:end]
            paths, weights, path_sizes,sample_time = cugraph.random_walks(
                graph,
                random_walks_type="uniform",
                start_vertices=sub_slicing,
                max_depth=walk_length,
                legacy_result_type=True,
            )
            epoch_sample_time+=sample_time
        torch.cuda.synchronize()
        end_time = time.time()
        time_list.append(end_time - start_time)
        sample_list.append(epoch_sample_time)
        print(
            "Run {} seeds, {} times, epoch run time: {:.6f}, epoch sample time: {:.6f}ms".format(
                len(seeds), batchnum, time_list[-1] * 1000, sample_list[-1] * 1000
            )
        )
    print("avg epoch time:", np.mean(sample_list[1:]) * 1000)
<<<<<<< HEAD
=======
    with open("../outputs/result.csv", "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        # system name, dataset, sampling time, mem peak
        log_info = ["cugraph", args.dataset, np.mean(sample_list[1:]), "rw"]
        writer.writerow(log_info)
        print(f"result writen to ../outputs/result.csv")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="livejournal",
    choices=["producrs", "livejournal"],
    help="which dataset to load for training",
)
args = parser.parse_args()
if args.dataset=="livejournal":
    dataset = load_livejournal()
>>>>>>> 88032ccd267c2bc12b1636703623dd3b7fbedd47

else:
    dataset = load_ogb("ogbn-products")

dgl_graph = dataset[0]
train_id = dataset[4]["train"]
train_id = train_id.cpu().numpy()
index = np.random.permutation(train_id.shape[0])
train_id = cudf.Series(train_id[index])
dgl_graph = dgl_graph.to("cuda")
g_cugraph = dgl_graph.to_cugraph()
del dgl_graph
print("Timing random walks")
batchsize = 1024
walk_len = 80
batchnum = int((train_id.shape[0] + batchsize - 1) / batchsize)
time_randomwalk(g_cugraph, train_id, batchsize, walk_len, batchnum)
# coo_matrix = sp.load_npz("/home/ubuntu/data/friendster/friendster_adj.npz")
# g = dgl.from_scipy(coo_matrix)
# g = g.formats("csc")
