import torch
import gs
import numpy as np


def plain(A, fanouts, seeds):
    blocks = []
    output_nodes = seeds
    for fanout in fanouts:
        subg, _ = A._graph._CAPI_Slicing(seeds, 1, gs._CSC, gs._CSC)
        subg, _ = subg._CAPI_Sampling(1, fanout, False, gs._CSC, gs._CSC)
        frontier = subg._CAPI_GetCSCIndices()
        blocks.append(frontier)
        seeds = frontier
    input_nodes = seeds
    return input_nodes, output_nodes, blocks


def fusion(A, fanouts, seeds):
    blocks = []
    output_nodes = seeds
    for fanout in fanouts:
        subg, _ = A._graph._CAPI_SlicingSampling(1, seeds, fanout, False, gs._CSC, gs._CSC)
        frontier = subg._CAPI_GetCSCIndices()
        blocks.append(frontier)
        seeds = frontier
    input_nodes = seeds
    return input_nodes, output_nodes, blocks


def batch_fusion(A, fanouts, seeds, seeds_ptr):
    ptrts, indts, blocks = [], [], []
    for layer, fanout in enumerate(fanouts):
        subg, _ = A._graph._CAPI_BatchFusedSlicingSampling(seeds, seeds_ptr, fanout, False)
        indptr, _ = subg._CAPI_BatchGetCSCIndptr()
        indices, indices_ptr = subg._CAPI_BatchGetCSCIndices()
        ptrt = torch.ops.gs_ops._CAPI_BatchIndptrSplitByOffset(indptr, seeds_ptr)
        indt = torch.ops.gs_ops._CAPI_BatchSplitByOffset(indices, indices_ptr)
        ptrts.append(ptrt)
        indts.append(indt)
        seeds, seeds_ptr = indices, indices_ptr
    return ptrts, indts, blocks
