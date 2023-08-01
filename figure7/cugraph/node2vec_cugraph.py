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
import csv 
import argparse
import sys 
sys.path.append("..") 
from load_graph_utils import load_ogbn_products,load_livejournal

def time_randomwalk(graph, seeds, batchsize, walk_length, batchnum):
    """
    Test cost time of random walk
    """
    runs = 6
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
            paths, weights, path_sizes,sample_time = cugraph.node2vec(
                graph,
                start_vertices=sub_slicing,
                max_depth=walk_length,
                compress_result=True,
                p=2.0,
                q=0.5,
            )
            epoch_sample_time+=sample_time
        torch.cuda.synchronize()
        end_time = time.time()
        time_list.append(end_time - start_time)
        sample_list.append(epoch_sample_time)
        print(
            "Run {} seeds, {} times, epoch run time: {:.6f} ms, epoch sample time: {:.6f} ms".format(
                len(seeds), batchnum, time_list[-1] * 1000,sample_list[-1] * 1000
            )
        )
    print("avg epoch time:", np.mean(sample_list[1:]) * 1000)
    with open("../outputs/result.csv", "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        # system name, dataset, sampling time, mem peak
        log_info = ["cuGraph", args.dataset, np.mean(sample_list[1:]), "node2vec"]
        writer.writerow(log_info)
        print(f"result writen to ../outputs/result.csv")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="livejournal",
    choices=["products", "livejournal"],
    help="which dataset to load for training",
)
args = parser.parse_args()
if args.dataset=="livejournal":
    dataset = load_livejournal()
else:
    dataset = load_ogbn_products()


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
time_randomwalk(g_cugraph, train_id, batchsize, 80, batchnum)
# coo_matrix = sp.load_npz("/home/ubuntu/data/friendster/friendster_adj.npz")
# g = dgl.from_scipy(coo_matrix)
# g = g.formats("csc")
