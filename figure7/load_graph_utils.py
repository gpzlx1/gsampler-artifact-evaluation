from ogb.nodeproppred import DglNodePropPredDataset
import torch
import dgl
import scipy.sparse as sp
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
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
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
    train_id = torch.load("/home/ubuntu/dataset/livejournal/livejournal_trainid.pt")
    splitted_idx = dict()
    splitted_idx['train']=train_id
    coo_matrix = sp.load_npz("/home/ubuntu/dataset/livejournal/livejournal_adj.npz")
    g = dgl.from_scipy(coo_matrix)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g=g.long()
    return g, None, None, None, splitted_idx

def load_friendster():
    train_id = torch.load("/home/ubuntu/dataset/friendster/friendster_trainid.pt")
    splitted_idx = dict()
    splitted_idx['train']=train_id
    bin_path = "/home/ubuntu/dataset/friendster/friendster_adj.bin"
    g_list, _ = dgl.load_graphs(bin_path)
    g = g_list[0]
    print("graph loaded")
    g=g.long()
    return g, None,None,None,splitted_idx