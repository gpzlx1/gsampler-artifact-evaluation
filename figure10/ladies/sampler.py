import torch
import gs
import numpy as np
from typing import List


def ladies_sampler(A: gs.Matrix, seeds: torch.Tensor, fanouts: List):
    input_node = seeds
    ret = []
    for K in fanouts:
        subA = A[:, seeds]
        prob = subA.sum("w", axis=1)
        sampleA, select_index = subA.collective_sampling(K, prob, False)
        sampleA = sampleA.div("w", prob[select_index], axis=1)
        out = sampleA.sum("w", axis=0)
        sampleA = sampleA.div("w", out, axis=0)
        seeds = sampleA.all_nodes()
        ret.append(sampleA)
    output_node = seeds
    return input_node, output_node, ret


def batch_ladies_sampler(A: gs.BatchMatrix, fanouts: List, seeds: torch.Tensor, seeds_ptr: torch.Tensor):
    ret = []
    for K in fanouts:
        subA = A[:, seeds::seeds_ptr]
        subA.edata["p"] = subA.edata["w"]**2
        prob = subA.sum("p", axis=1)
        neighbors, probs_ptr = subA.all_rows()
        sampleA, select_index = subA.collective_sampling(
            K, prob, probs_ptr, False)
        sampleA = sampleA.div("w", prob[select_index], axis=1)
        out = sampleA.sum("w", axis=0)
        sampleA = sampleA.div("w", out, axis=0)
        seeds, seeds_ptr = sampleA.all_nodes()
        ret.append(sampleA.to_dgl_block())

    return ret
