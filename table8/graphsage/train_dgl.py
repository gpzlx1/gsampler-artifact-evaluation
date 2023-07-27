import torch
from dgl.dataloading import DataLoader, NeighborSampler
from dgl.utils import gather_pinned_tensor_rows
import time
import argparse
from ctypes import *
from ctypes.util import *
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from gs.utils.load_graph import *
from model import *
import csv


def train_dgl(dataset, args):
    g, features, labels, n_classes, splitted_idx = dataset
    g = g.formats('csc')
    train_nid, val_nid, _ = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    features, labels = features.to('cuda'), labels.to('cuda')

    g = g.to('cpu')
    train_nid, val_nid = train_nid.to('cpu'), val_nid.to('cpu')
    num_layers = 3
    model = DGLSAGEModel(features.shape[1], 256, n_classes, num_layers).to('cuda')
    sampler = NeighborSampler([10, 25])
    train_dataloader = DataLoader(g, train_nid, sampler, batch_size=args.batchsize,
                                  shuffle=True,  drop_last=False, num_workers=args.num_workers)
    val_dataloader = DataLoader(g, val_nid, sampler, batch_size=args.batchsize,
                                shuffle=True, drop_last=False, num_workers=args.num_workers)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    static_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    print('memory allocated before training:', static_memory, 'GB')

    epoch_time = []
    cur_time = []
    acc_list = []
    start = time.time()
    for epoch in range(args.num_epoch):
        sampling_time = 0
        torch.cuda.synchronize()
        tic = time.time()
        model.train()
        sampling_start = time.time()
        for step, (input_nodes, output_nodes, blocks) in enumerate(tqdm(train_dataloader)):
            input_nodes, output_nodes = input_nodes.cuda(), output_nodes.cuda()
            blocks = [block.to('cuda') for block in blocks]
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
                blocks = [block.to('cuda') for block in blocks]
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

        print("Epoch {:05d} | Val ACC {:.4f} | Epoch Time {:.4f} s | Accumulated Time {:.4f} s | Sampling Time: {:.4f} s".format(
            epoch, acc, epoch_time[-1], cur_time[-1], sampling_time))

    torch.cuda.synchronize()
    total_time = time.time() - start

    print('Total Elapse Time:', total_time)
    print('Average Epoch Time:', np.mean(epoch_time[3:]))
    s5 = pd.Series(cur_time, name='cumulated time/s')
    s1 = pd.Series(acc_list, name='acc')
    s2 = pd.Series(epoch_time, name='time/s')
    s3 = pd.Series([total_time], name='total time/s')
    s4 = pd.Series([static_memory], name='static mem/GB')
    df = pd.concat([s5, s1, s2, s3, s4], axis=1)
    df.to_csv('outputs/data/graphsage_dgl_{}_{}.csv'.format(args.dataset,
              time.ctime().replace(' ', '_')), index=False)
    
    with open("../outputs/result.csv", "a") as f:
        writer = csv.writer(f, lineterminator="\n")
        # system name, dataset, total time, acc
        log_info = ["DGL", "GraphSAGE", f"Time: {round(total_time, 2)} s", f"Accuracy: {round(np.max(acc_list) * 100, 2)}"]
        writer.writerow(log_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'],
                        help="Training model on gpu or cpu")
    parser.add_argument("--dataset", default='products', choices=['reddit', 'products', 'papers100m'],
                        help="which dataset to load for training")
    parser.add_argument("--batchsize", type=int, default=512,
                        help="batch size for training")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="numbers of workers for sampling, must be 0 when gpu or uva is used")
    parser.add_argument("--num_epoch", type=int, default=100,
                        help="numbers of epoch in training")
    args = parser.parse_args()
    print(args)
    if args.dataset == 'reddit':
        dataset = load_reddit()
    elif args.dataset == 'products':
        dataset = load_ogb('ogbn-products', "/home/ubuntu/dataset")
    elif args.dataset == 'papers100m':
        dataset = load_ogb('ogbn-papers100M', "/home/ubuntu/dataset")
    print(dataset[0])
    train_dgl(dataset, args)