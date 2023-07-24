import torch
import torch.nn.functional as F
from dgl.dataloading import DataLoader
from dgl.utils import gather_pinned_tensor_rows
from tqdm import tqdm
import argparse
import numpy as np
import time
import pandas as pd
from gs.utils.load_graph import *
from model import *


class LADIESSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts, weight="w", out_weight="w", replace=False, W=None):
        super().__init__()
        self.fanouts = fanouts
        self.edge_weight = weight
        self.output_weight = out_weight
        self.replace = replace
        self.return_eids = False
        self.W = W

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        blocks = []
        output_nodes = seed_nodes
        for fanout in self.fanouts:
            subg = dgl.in_subgraph(g, seed_nodes)
            # layer-wise sample
            edges = subg.edges()
            nodes = torch.unique(edges[0])
            num_pick = np.min([nodes.shape[0], fanout])
            reversed_subg = dgl.reverse(subg, copy_edata=True)
            weight = self.W[reversed_subg.edata[dgl.EID]]
            probs = dgl.ops.copy_e_sum(reversed_subg, weight**2)
            node_probs = probs[nodes]
            idx = torch.multinomial(node_probs, num_pick, replacement=False)
            selected = nodes[idx]
            ################
            selected = torch.cat((seed_nodes, selected)).unique()
            subg = dgl.out_subgraph(subg, selected)
            weight = weight[subg.edata[dgl.EID]]
            W_tilde = dgl.ops.e_div_u(subg, weight, probs)
            W_tilde_sum = dgl.ops.copy_e_sum(subg, W_tilde)
            W_tilde = dgl.ops.e_div_v(subg, W_tilde, W_tilde_sum)
            block = dgl.to_block(subg, seed_nodes)
            block.edata[self.output_weight] = W_tilde[block.edata[dgl.EID]]
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        input_nodes = seed_nodes
        return input_nodes, output_nodes, blocks


def train(dataset, args):
    device = args.device
    g, features, labels, n_classes, splitted_idx = dataset
    train_nid, val_nid = splitted_idx["train"], splitted_idx["valid"]
    g = g.to(device)
    train_nid, val_nid = train_nid.to(device), val_nid.to(device)
    features, labels = features.to("cuda"), labels.to("cuda")
    W = normalized_laplacian_edata(g)
    g = g.formats("csc")

    fanout = [4000, 4000, 4000]
    model = SAGEModel(features.shape[1], 256, n_classes, len(fanout)).to("cuda")
    sampler = LADIESSampler(fanout, weight="weight", out_weight="w", replace=False, W=W)
    train_dataloader = DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        g,
        val_nid,
        sampler,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    static_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    print("memory allocated before training:", static_memory, "GB")

    epoch_time = []
    acc_list = []
    cur_time = []
    start = time.time()
    for epoch in range(args.num_epoch):
        sampling_time = 0
        torch.cuda.synchronize()
        tic = time.time()
        model.train()
        sampling_start = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(tqdm(train_dataloader)):
            input_nodes, output_nodes = input_nodes.cuda(), output_nodes.cuda()
            blocks = [block.to("cuda") for block in blocks]
            sampling_time += time.time() - sampling_start

            x = features[input_nodes]
            y = labels[output_nodes]
            y_hat = model(blocks, x)
            is_labeled = y == y
            y = y[is_labeled].long()
            y_hat = y_hat[is_labeled]
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            sampling_start = time.time()

        model.eval()
        val_pred = []
        val_labels = []
        with torch.no_grad():
            sampling_start = time.time()
            for it, (input_nodes, output_nodes, blocks) in enumerate(tqdm(val_dataloader)):
                input_nodes, output_nodes = input_nodes.cuda(), output_nodes.cuda()
                blocks = [block.to("cuda") for block in blocks]
                sampling_time += time.time() - sampling_start

                x = features[input_nodes]
                y = labels[output_nodes]
                y_pred = model(blocks, x)
                val_pred.append(y_pred)
                val_labels.append(y)

                sampling_start = time.time()
        pred = torch.cat(val_pred)
        label = torch.cat(val_labels)
        acc = (pred.argmax(1) == label).float().mean().item()
        acc_list.append(acc)

        torch.cuda.synchronize()
        end = time.time()
        cur_time.append(end - start)
        epoch_time.append(end - tic)

        print(
            "Epoch {:05d} | Val ACC {:.4f} | Epoch Time {:.4f} s | Accumulated Time {:.4f} s | Sampling Time: {:.4f} s".format(
                epoch, acc, epoch_time[-1], cur_time[-1], sampling_time
            )
        )

    torch.cuda.synchronize()
    total_time = time.time() - start

    print("Total Elapse Time:", total_time)
    print("Average Epoch Time:", np.mean(epoch_time[3:]))
    s5 = pd.Series(cur_time, name="cumulated time/s")
    s1 = pd.Series(acc_list, name="acc")
    s2 = pd.Series(epoch_time, name="time/s")
    s3 = pd.Series([total_time], name="total time/s")
    s4 = pd.Series([static_memory], name="static mem/GB")
    df = pd.concat([s5, s1, s2, s3, s4], axis=1)
    df.to_csv(
        "outputs/data/ladies_dgl_{}_{}.csv".format(args.dataset, time.ctime().replace(" ", "_")),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Training model on gpu or cpu",
    )
    parser.add_argument(
        "--dataset",
        default="products",
        choices=["reddit", "products", "papers100m"],
        help="which dataset to load for training",
    )
    parser.add_argument("--batchsize", type=int, default=512, help="batch size for training")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="numbers of workers for sampling, must be 0 when gpu or uva is used",
    )
    parser.add_argument("--num_epoch", type=int, default=100, help="numbers of epoch in training")
    args = parser.parse_args()
    print(args)
    if args.dataset == "reddit":
        dataset = load_reddit()
    elif args.dataset == "products":
        dataset = load_ogb("ogbn-products", "/home/ubuntu/dataset")
    elif args.dataset == "papers100m":
        dataset = load_ogb("ogbn-papers100M", "/home/ubuntu/dataset")
    print(dataset[0])
    train(dataset, args)
