import torch
import gs
import numpy as np


def w_o_relabel(P: gs.Matrix, fanouts, seeds):
    graph = P._graph
    output_node = seeds
    ret = []
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, gs._CSC, gs._COO, False)
        probs = subg._CAPI_sum(1, 2, gs._CSR)
        row_nodes = subg._CAPI_get_valid_rows()
        node_probs = probs[row_nodes]
        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(row_nodes, node_probs, fanout, False)
        nodes = torch.cat((seeds, selected)).unique()  # add self-loop
        subg = subg._CAPI_slicing(nodes, 1, gs._CSR, gs._COO, False)
        subg = subg._CAPI_divide(probs[nodes], 1, gs._COO)
        _sum = subg._CAPI_sum(0, 1, gs._CSC)
        subg = subg._CAPI_divide(_sum, 0, gs._COO)
        (
            unique_tensor,
            num_row,
            num_col,
            format_tensor1,
            format_tensor2,
            e_ids,
            format,
        ) = subg._CAPI_relabel()
        seeds = unique_tensor
    input_node = seeds
    return input_node, output_node, ret


def w_o_relabel_fusion(P: gs.Matrix, fanouts, seeds):
    graph = P._graph
    output_node = seeds
    ret = []
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, gs._CSC, gs._COO, False)
        probs = subg._CAPI_sum(1, 2, gs._CSR)
        row_nodes = subg._CAPI_get_valid_rows()
        node_probs = probs[row_nodes]
        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(row_nodes, node_probs, fanout, False)
        nodes = torch.cat((seeds, selected)).unique()  # add self-loop
        subg = subg._CAPI_slicing(nodes, 1, gs._CSR, gs._COO, False)
        subg, _sum = subg._CAPI_e_div_u_sum(probs[nodes])
        subg = subg._CAPI_divide(_sum, 0, gs._COO)
        (
            unique_tensor,
            num_row,
            num_col,
            format_tensor1,
            format_tensor2,
            e_ids,
            format,
        ) = subg._CAPI_relabel()
        seeds = unique_tensor
    input_node = seeds
    return input_node, output_node, ret


def w_o_relabel_fusion_selection(P: gs.Matrix, fanouts, seeds):
    graph = P._graph
    output_node = seeds
    ret = []
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, gs._CSC, gs._COO, False)
        probs = subg._CAPI_sum(1, 2, gs._COO)
        row_nodes = subg._CAPI_get_valid_rows()
        node_probs = probs[row_nodes]
        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(row_nodes, node_probs, fanout, False)
        nodes = torch.cat((seeds, selected)).unique()  # add self-loop
        subg = subg._CAPI_slicing(nodes, 1, gs._COO, gs._COO, False)
        subg, _sum = subg._CAPI_e_div_u_sum(probs[nodes])
        subg = subg._CAPI_divide(_sum, 0, gs._COO)
        (
            unique_tensor,
            num_row,
            num_col,
            format_tensor1,
            format_tensor2,
            e_ids,
            format,
        ) = subg._CAPI_relabel()
        seeds = unique_tensor
    input_node = seeds
    return input_node, output_node, ret


def w_relabel(P: gs.Matrix, fanouts, seeds):
    graph = P._graph
    output_node = seeds
    ret = []
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, gs._CSC, gs._COO, True)
        probs = subg._CAPI_sum(1, 2, gs._CSR)
        num_pick = np.min([probs.numel(), fanout])
        idx = torch.multinomial(probs, num_pick, replacement=False)
        relabel_seeds_nodes = torch.ops.gs_ops.index_search(subg._CAPI_get_rows(), seeds)
        nodes = torch.cat((relabel_seeds_nodes, idx)).unique()  # add self-loop
        subg = subg._CAPI_slicing(nodes, 1, gs._CSR, gs._COO, False)
        subg = subg._CAPI_divide(probs[nodes], 1, gs._COO)
        _sum = subg._CAPI_sum(0, 1, gs._CSC)
        subg = subg._CAPI_divide(_sum, 0, gs._COO)
        (
            unique_tensor,
            num_row,
            num_col,
            format_tensor1,
            format_tensor2,
            e_ids,
            format,
        ) = subg._CAPI_relabel()
        seeds = unique_tensor
    input_node = seeds
    return input_node, output_node, ret


def w_relabel_fusion_selection(P: gs.Matrix, fanouts, seeds):
    graph = P._graph
    output_node = seeds
    ret = []
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, gs._CSC, gs._COO, True)
        probs = subg._CAPI_sum(1, 2, gs._COO)
        num_pick = np.min([probs.numel(), fanout])
        idx = torch.multinomial(probs, num_pick, replacement=False)
        relabel_seeds_nodes = torch.ops.gs_ops.index_search(subg._CAPI_get_rows(), seeds)
        nodes = torch.cat((relabel_seeds_nodes, idx)).unique()  # add self-loop
        subg = subg._CAPI_slicing(nodes, 1, gs._COO, gs._COO, False)
        subg, _sum = subg._CAPI_e_div_u_sum(probs[nodes])
        subg = subg._CAPI_divide(_sum, 0, gs._COO)
        (
            unique_tensor,
            num_row,
            num_col,
            format_tensor1,
            format_tensor2,
            e_ids,
            format,
        ) = subg._CAPI_relabel()
        seeds = unique_tensor
    input_node = seeds
    return input_node, output_node, ret


def batching_w_o_relabel(P: gs.Matrix, fanouts, seeds, seeds_ptr):
    num_batches = seeds_ptr.numel() - 1
    graph = P._graph
    output_node = seeds
    blocks = []
    encoding_size = graph._CAPI_get_num_rows()
    for fanout in fanouts:
        # (batchID * num_nodes) * nodeID
        subg, _ = graph._CAPI_batch_slicing(seeds, seeds_ptr, 0, gs._CSC, gs._COO, False, True)
        probs = subg._CAPI_sum(1, 2, gs._CSR)

        neighbors = torch.unique(subg._CAPI_get_coo_rows(False))
        # int(nodeID / num_nodes)
        node_probs = probs[neighbors]
        neighbors_ptr, _ = torch.ops.gs_ops.GetBatchOffsets(neighbors, num_batches, encoding_size)
        idx, _ = torch.ops.gs_ops.batch_list_sampling_with_probs(node_probs, fanout, False, neighbors_ptr)
        selected = neighbors[idx]

        nodes = torch.cat((subg._CAPI_get_cols(), selected)).unique()
        subg = subg._CAPI_slicing(nodes, 1, gs._CSR, gs._COO, False)  # Row Slicing
        subg = subg._CAPI_divide(probs[nodes], 1, gs._COO)
        _sum = subg._CAPI_sum(0, 1, gs._CSC)
        subg = subg._CAPI_divide(_sum, 0, gs._COO)

        encoded_coo_row = subg._CAPI_get_rows()[subg._CAPI_get_coo_rows(False)]
        # nodeID - int(nodeID / num_nodes) * num_nodes
        coo_ptr, coo_row = torch.ops.gs_ops.GetBatchOffsets(encoded_coo_row, num_batches, encoding_size)
        coo_col = seeds[subg._CAPI_get_coo_cols(False)]
        (
            unique_tensor,
            unique_tensor_ptr,
            sub_coo_row,
            sub_coo_col,
            sub_coo_ptr,
        ) = torch.ops.gs_ops.BatchCOORelabel(seeds, seeds_ptr, coo_col, coo_row, coo_ptr)
        seedst = torch.ops.gs_ops.SplitByOffset(seeds, seeds_ptr)
        unit = torch.ops.gs_ops.SplitByOffset(unique_tensor, unique_tensor_ptr)
        colt = torch.ops.gs_ops.SplitByOffset(sub_coo_col, sub_coo_ptr)
        rowt = torch.ops.gs_ops.SplitByOffset(sub_coo_row, sub_coo_ptr)
        blocks.insert(0, (seedst, unit, colt, rowt))

        seeds, seeds_ptr = unique_tensor, unique_tensor_ptr
    input_node = seeds
    return input_node, output_node, blocks


# def batching_w_o_relabel_fusion(P: gs.Matrix, fanouts, seeds, seeds_ptr):
#     num_batches = seeds_ptr.numel() - 1
#     graph = P._graph
#     output_node = seeds
#     blocks = []
#     encoding_size = graph._CAPI_get_num_rows()
#     for fanout in fanouts:
#         # (batchID * num_nodes) * nodeID
#         subg, _ = graph._CAPI_batch_slicing(
#             seeds, seeds_ptr, 0, gs._CSC, gs._COO, False, True
#         )
#         probs = subg._CAPI_sum(1, 2, gs._CSR)

#         neighbors = torch.unique(subg._CAPI_get_coo_rows(False))
#         # int(nodeID / num_nodes)
#         node_probs = probs[neighbors]
#         neighbors_ptr, _ = torch.ops.gs_ops.GetBatchOffsets(
#             neighbors, num_batches, encoding_size
#         )
#         idx, _ = torch.ops.gs_ops.batch_list_sampling_with_probs(
#             node_probs, fanout, False, neighbors_ptr
#         )
#         selected = neighbors[idx]

#         nodes = torch.cat((subg._CAPI_get_cols(), selected)).unique()
#         subg = subg._CAPI_slicing(nodes, 1, gs._CSR, gs._COO, False)  # Row Slicing
#         subg = subg._CAPI_divide(probs[nodes], 1, gs._COO)
#         subg = subg._CAPI_normalize(0, gs._CSC)

#         encoded_coo_row = subg._CAPI_get_rows()[subg._CAPI_get_coo_rows(False)]
#         # nodeID - int(nodeID / num_nodes) * num_nodes
#         coo_ptr, coo_row = torch.ops.gs_ops.GetBatchOffsets(
#             encoded_coo_row, num_batches, encoding_size
#         )
#         coo_col = seeds[subg._CAPI_get_coo_cols(False)]
#         (
#             unique_tensor,
#             unique_tensor_ptr,
#             sub_coo_row,
#             sub_coo_col,
#             sub_coo_ptr,
#         ) = torch.ops.gs_ops.BatchCOORelabel(
#             seeds, seeds_ptr, coo_col, coo_row, coo_ptr
#         )
#         seedst = torch.ops.gs_ops.SplitByOffset(seeds, seeds_ptr)
#         unit = torch.ops.gs_ops.SplitByOffset(unique_tensor, unique_tensor_ptr)
#         colt = torch.ops.gs_ops.SplitByOffset(sub_coo_col, sub_coo_ptr)
#         rowt = torch.ops.gs_ops.SplitByOffset(sub_coo_row, sub_coo_ptr)
#         blocks.insert(0, (seedst, unit, colt, rowt))

#         seeds, seeds_ptr = unique_tensor, unique_tensor_ptr
#     input_node = seeds
#     return input_node, output_node, blocks


def batching_w_o_relabel_fusion_selection(P: gs.Matrix, fanouts, seeds, seeds_ptr):
    num_batches = seeds_ptr.numel() - 1
    graph = P._graph
    output_node = seeds
    blocks = []
    encoding_size = graph._CAPI_get_num_rows()
    for fanout in fanouts:
        # (batchID * num_nodes) * nodeID
        subg, _ = graph._CAPI_batch_slicing(seeds, seeds_ptr, 0, gs._CSC, gs._COO, False, True)
        probs = subg._CAPI_sum(1, 2, gs._COO)

        neighbors = torch.unique(subg._CAPI_get_coo_rows(False))
        # int(nodeID / num_nodes)
        node_probs = probs[neighbors]
        neighbors_ptr, _ = torch.ops.gs_ops.GetBatchOffsets(neighbors, num_batches, encoding_size)
        idx, _ = torch.ops.gs_ops.batch_list_sampling_with_probs(node_probs, fanout, False, neighbors_ptr)
        selected = neighbors[idx]

        nodes = torch.cat((subg._CAPI_get_cols(), selected)).unique()
        subg = subg._CAPI_slicing(nodes, 1, gs._COO, gs._COO, False)  # Row Slicing
        subg, _sum = subg._CAPI_e_div_u_sum(probs[nodes])
        subg = subg._CAPI_divide(_sum, 0, gs._COO)

        encoded_coo_row = subg._CAPI_get_rows()[subg._CAPI_get_coo_rows(False)]
        # nodeID - int(nodeID / num_nodes) * num_nodes
        coo_ptr, coo_row = torch.ops.gs_ops.GetBatchOffsets(encoded_coo_row, num_batches, encoding_size)
        coo_col = seeds[subg._CAPI_get_coo_cols(False)]
        (
            unique_tensor,
            unique_tensor_ptr,
            sub_coo_row,
            sub_coo_col,
            sub_coo_ptr,
        ) = torch.ops.gs_ops.BatchCOORelabel(seeds, seeds_ptr, coo_col, coo_row, coo_ptr)
        seedst = torch.ops.gs_ops.SplitByOffset(seeds, seeds_ptr)
        unit = torch.ops.gs_ops.SplitByOffset(unique_tensor, unique_tensor_ptr)
        colt = torch.ops.gs_ops.SplitByOffset(sub_coo_col, sub_coo_ptr)
        rowt = torch.ops.gs_ops.SplitByOffset(sub_coo_row, sub_coo_ptr)
        blocks.insert(0, (seedst, unit, colt, rowt))

        seeds, seeds_ptr = unique_tensor, unique_tensor_ptr
    input_node = seeds
    return input_node, output_node, blocks


def batching_w_relabel(P: gs.Matrix, fanouts, seeds, seeds_ptr):
    num_batches = seeds_ptr.numel() - 1
    graph = P._graph
    output_node = seeds
    blocks = []
    encoding_size = graph._CAPI_get_num_rows()
    for fanout in fanouts:
        subg, _ = graph._CAPI_batch_slicing(seeds, seeds_ptr, 0, gs._CSC, gs._COO, True, True)
        probs = subg._CAPI_sum(1, 2, gs._CSR)
        num_pick = np.min([probs.numel(), fanout])

        # int(nodeID / num_nodes)
        row_ptr, _ = torch.ops.gs_ops.GetBatchOffsets(subg._CAPI_get_rows(), num_batches, encoding_size)
        selected, _ = torch.ops.gs_ops.batch_list_sampling_with_probs(probs, num_pick, False, row_ptr)

        relabel_seeds_nodes = torch.ops.gs_ops.index_search(subg._CAPI_get_rows(), subg._CAPI_get_cols())
        nodes = torch.cat((relabel_seeds_nodes, selected)).unique()
        subg = subg._CAPI_slicing(nodes, 1, gs._CSR, gs._COO, False)  # Row Slicing
        subg = subg._CAPI_divide(probs[nodes], 1, gs._COO)
        _sum = subg._CAPI_sum(0, 1, gs._CSC)
        subg = subg._CAPI_divide(_sum, 0, gs._COO)

        encoded_coo_row = subg._CAPI_get_rows()[subg._CAPI_get_coo_rows(False)]
        # int(nodeID / num_nodes)
        coo_ptr, coo_row = torch.ops.gs_ops.GetBatchOffsets(encoded_coo_row, num_batches, encoding_size)
        coo_col = seeds[subg._CAPI_get_coo_cols(False)]
        (
            unique_tensor,
            unique_tensor_ptr,
            sub_coo_row,
            sub_coo_col,
            sub_coo_ptr,
        ) = torch.ops.gs_ops.BatchCOORelabel(seeds, seeds_ptr, coo_col, coo_row, coo_ptr)
        seedst = torch.ops.gs_ops.SplitByOffset(seeds, seeds_ptr)
        unit = torch.ops.gs_ops.SplitByOffset(unique_tensor, unique_tensor_ptr)
        colt = torch.ops.gs_ops.SplitByOffset(sub_coo_col, coo_ptr)
        rowt = torch.ops.gs_ops.SplitByOffset(sub_coo_row, coo_ptr)
        eweight = torch.ops.gs_ops.SplitByOffset(subg._CAPI_get_data("default"), coo_ptr)
        blocks.insert(0, (seedst, unit, colt, rowt, eweight))

        seeds, seeds_ptr = unique_tensor, unique_tensor_ptr
    input_node = seeds
    return input_node, output_node, blocks


# def batching_w_relabel_fusion(P: gs.Matrix, fanouts, seeds, seeds_ptr):
#     num_batches = seeds_ptr.numel() - 1
#     graph = P._graph
#     output_node = seeds
#     blocks = []
#     encoding_size = graph._CAPI_get_num_rows()
#     for fanout in fanouts:
#         subg, _ = graph._CAPI_batch_slicing(
#             seeds, seeds_ptr, 0, gs._CSC, gs._COO, True, True
#         )
#         probs = subg._CAPI_sum(1, 2, gs._CSR)
#         num_pick = np.min([probs.numel(), fanout])

#         # int(nodeID / num_nodes)
#         row_ptr, _ = torch.ops.gs_ops.GetBatchOffsets(
#             subg._CAPI_get_rows(), num_batches, encoding_size
#         )
#         selected, _ = torch.ops.gs_ops.batch_list_sampling_with_probs(
#             probs, num_pick, False, row_ptr
#         )

#         relabel_seeds_nodes = torch.ops.gs_ops.index_search(
#             subg._CAPI_get_rows(), subg._CAPI_get_cols()
#         )
#         nodes = torch.cat((relabel_seeds_nodes, selected)).unique()
#         subg = subg._CAPI_slicing(nodes, 1, gs._CSR, gs._COO, False)  # Row Slicing
#         subg = subg._CAPI_divide(probs[nodes], 1, gs._COO)
#         subg = subg._CAPI_normalize(0, gs._CSC)

#         encoded_coo_row = subg._CAPI_get_rows()[subg._CAPI_get_coo_rows(False)]
#         # int(nodeID / num_nodes)
#         coo_ptr, coo_row = torch.ops.gs_ops.GetBatchOffsets(
#             encoded_coo_row, num_batches, encoding_size
#         )
#         coo_col = seeds[subg._CAPI_get_coo_cols(False)]
#         (
#             unique_tensor,
#             unique_tensor_ptr,
#             sub_coo_row,
#             sub_coo_col,
#             sub_coo_ptr,
#         ) = torch.ops.gs_ops.BatchCOORelabel(
#             seeds, seeds_ptr, coo_col, coo_row, coo_ptr
#         )
#         seedst = torch.ops.gs_ops.SplitByOffset(seeds, seeds_ptr)
#         unit = torch.ops.gs_ops.SplitByOffset(unique_tensor, unique_tensor_ptr)
#         colt = torch.ops.gs_ops.SplitByOffset(sub_coo_col, coo_ptr)
#         rowt = torch.ops.gs_ops.SplitByOffset(sub_coo_row, coo_ptr)
#         eweight = torch.ops.gs_ops.SplitByOffset(
#             subg._CAPI_get_data("default"), coo_ptr
#         )
#         blocks.insert(0, (seedst, unit, colt, rowt, eweight))

#         seeds, seeds_ptr = unique_tensor, unique_tensor_ptr
#     input_node = seeds
#     return input_node, output_node, blocks


def batching_w_relabel_fusion_selection(P: gs.Matrix, fanouts, seeds, seeds_ptr):
    num_batches = seeds_ptr.numel() - 1
    graph = P._graph
    output_node = seeds
    blocks = []
    encoding_size = graph._CAPI_get_num_rows()
    for fanout in fanouts:
        subg, _ = graph._CAPI_batch_slicing(seeds, seeds_ptr, 0, gs._CSC, gs._COO, True, True)
        probs = subg._CAPI_sum(1, 2, gs._COO)
        num_pick = np.min([probs.numel(), fanout])

        # int(nodeID / num_nodes)
        row_ptr, _ = torch.ops.gs_ops.GetBatchOffsets(subg._CAPI_get_rows(), num_batches, encoding_size)
        selected, _ = torch.ops.gs_ops.batch_list_sampling_with_probs(probs, num_pick, False, row_ptr)

        relabel_seeds_nodes = torch.ops.gs_ops.index_search(subg._CAPI_get_rows(), subg._CAPI_get_cols())
        nodes = torch.cat((relabel_seeds_nodes, selected)).unique()
        subg = subg._CAPI_slicing(nodes, 1, gs._COO, gs._COO, False)  # Row Slicing
        subg, _sum = subg._CAPI_e_div_u_sum(probs[nodes])
        subg = subg._CAPI_divide(_sum, 0, gs._COO)

        encoded_coo_row = subg._CAPI_get_rows()[subg._CAPI_get_coo_rows(False)]
        # int(nodeID / num_nodes)
        coo_ptr, coo_row = torch.ops.gs_ops.GetBatchOffsets(encoded_coo_row, num_batches, encoding_size)
        coo_col = seeds[subg._CAPI_get_coo_cols(False)]
        (
            unique_tensor,
            unique_tensor_ptr,
            sub_coo_row,
            sub_coo_col,
            sub_coo_ptr,
        ) = torch.ops.gs_ops.BatchCOORelabel(seeds, seeds_ptr, coo_col, coo_row, coo_ptr)
        seedst = torch.ops.gs_ops.SplitByOffset(seeds, seeds_ptr)
        unit = torch.ops.gs_ops.SplitByOffset(unique_tensor, unique_tensor_ptr)
        colt = torch.ops.gs_ops.SplitByOffset(sub_coo_col, coo_ptr)
        rowt = torch.ops.gs_ops.SplitByOffset(sub_coo_row, coo_ptr)
        eweight = torch.ops.gs_ops.SplitByOffset(subg._CAPI_get_data("default"), coo_ptr)
        blocks.insert(0, (seedst, unit, colt, rowt, eweight))

        seeds, seeds_ptr = unique_tensor, unique_tensor_ptr
    input_node = seeds
    return input_node, output_node, blocks
