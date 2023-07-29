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
        subg, _ = A._graph._CAPI_SlicingSampling(1, seeds, fanout, False,
                                                 gs._CSC, gs._CSC)
        frontier = subg._CAPI_GetCSCIndices()
        blocks.append(frontier)
        seeds = frontier
    input_nodes = seeds
    return input_nodes, output_nodes, blocks


def batch_fusion(A, fanouts, seeds, seeds_ptr):
    ret = []
    for k in fanouts:
        subA = A[:, seeds::seeds_ptr]
        sampleA = subA.individual_sampling(k, None, False)
        seeds, seeds_ptr = sampleA.all_nodes()
        ret.append(sampleA)
    return ret
