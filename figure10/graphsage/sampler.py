import torch
import gs
import numpy as np


def plain(A, fanouts, seeds):
    blocks = []
    output_nodes = seeds
    for fanout in fanouts:
        subg = A._graph._CAPI_slicing(seeds, 0, gs._CSC, gs._CSC, False)
        subg = subg._CAPI_sampling(0, fanout, False, gs._CSC, gs._CSC)
        frontier = subg._CAPI_get_coo_rows(False)
        blocks.append(frontier)
        seeds = frontier
    input_nodes = seeds
    return input_nodes, output_nodes, blocks


def fusion(A, fanouts, seeds):
    blocks = []
    output_nodes = seeds
    for fanout in fanouts:
        subg = A._graph._CAPI_fused_columnwise_slicing_sampling(seeds, fanout, False)
        frontier = subg._CAPI_get_coo_rows(False)
        blocks.append(frontier)
        seeds = frontier
    input_nodes = seeds
    return input_nodes, output_nodes, blocks


def batch(A: gs.Graph, fanouts, seeds, seeds_ptr):
    ptrts, indts, blocks = [], [], []
    for layer, fanout in enumerate(fanouts):
        subg = A._graph._CAPI_slicing(seeds, 0, gs._CSC, gs._CSC, False)
        subg = subg._CAPI_sampling(0, fanout, False, gs._CSC, gs._CSC)
        indptr, indices, eids = subg._CAPI_get_csc()
        indices_ptr = indptr[seeds_ptr]
        ptrt = torch.ops.gs_ops.IndptrSplitByOffset(indptr, seeds_ptr)
        indt = torch.ops.gs_ops.SplitByOffset(indices, indices_ptr)
        ptrts.append(ptrt)
        indts.append(indt)
        seeds, seeds_ptr = indices, indices_ptr
    return ptrts, indts, blocks


def batch_fusion(A: gs.Graph, fanouts, seeds, seeds_ptr):
    ptrts, indts, blocks = [], [], []
    for layer, fanout in enumerate(fanouts):
        subg = A._graph._CAPI_fused_columnwise_slicing_sampling(seeds, fanout, False)
        indptr, indices, eids = subg._CAPI_get_csc()
        indices_ptr = indptr[seeds_ptr]
        ptrt = torch.ops.gs_ops.IndptrSplitByOffset(indptr, seeds_ptr)
        indt = torch.ops.gs_ops.SplitByOffset(indices, indices_ptr)
        ptrts.append(ptrt)
        indts.append(indt)
        seeds, seeds_ptr = indices, indices_ptr
    return ptrts, indts, blocks
