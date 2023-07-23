import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


def normalized_laplacian_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = "W"
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g_rev = dgl.reverse(g, copy_edata=True)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, "v"))
        g_rev.update_all(fn.copy_e(weight, weight), fn.sum(weight, "u"))
        g.ndata["u"] = g_rev.ndata["u"]
        g.apply_edges(
            lambda edges: {
                "w": edges.data[weight] / torch.sqrt(edges.src["u"] * edges.dst["v"])
            }
        )
        return g.edata["w"]


class GraphConv(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.W = nn.Linear(in_feats, n_classes)

    def forward(self, g, x, w):
        with g.local_scope():
            g.srcdata["x"] = self.W(x)
            g.edata["w"] = w
            g.update_all(fn.u_mul_e("x", "w", "m"), fn.sum("m", "y"))
            return g.dstdata["y"]


class GCNModel(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_feats, n_hidden))
        for i in range(n_layers - 2):
            self.convs.append(GraphConv(n_hidden, n_hidden))
        self.convs.append(GraphConv(n_hidden, n_classes))

    def forward(self, blocks, x):
        for i, (conv, block) in enumerate(zip(self.convs, blocks)):
            x = conv(block, x, block.edata["w"])
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x


class SAGEConv(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.W = nn.Linear(in_feats * 2, n_classes)

    def forward(self, g, x, w):
        with g.local_scope():
            g.srcdata["x"] = x
            g.dstdata["x"] = x[: g.number_of_dst_nodes()]
            # g.edata['w'] = w
            # g.update_all(fn.u_mul_e('x', 'w', 'm'), fn.sum('m', 'y'))
            g.update_all(fn.copy_u("x", "m"), fn.mean("m", "y"))
            h = torch.cat([g.dstdata["x"], g.dstdata["y"]], 1)
            return self.W(h)


class SAGEModel(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_feats, n_hidden))
        for i in range(n_layers - 2):
            self.convs.append(SAGEConv(n_hidden, n_hidden))
        self.convs.append(SAGEConv(n_hidden, n_classes))

    def forward(self, blocks, x):
        for i, (conv, block) in enumerate(zip(self.convs, blocks)):
            x = conv(block, x, block.edata["w"])
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x
