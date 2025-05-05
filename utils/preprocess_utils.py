# utils/preprocess_utils.py

import torch
from torch_geometric.utils import add_self_loops, degree
from torch_sparse import coalesce 

def compute_propagation_matrix(edge_index: torch.Tensor,
                               num_nodes: int,
                               add_self_loops_flag: bool = True,
                               dtype: torch.dtype = torch.float) -> torch.Tensor:

    device = edge_index.device 

    # 1. 添加自环 (如果需要)
    if add_self_loops_flag:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # 2. 计算边的权重: 1 / sqrt(deg(src) * deg(dst))
    row, col = edge_index[0], edge_index[1]
    # 计算每个节点的度数
    deg = degree(col, num_nodes=num_nodes, dtype=dtype) 
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

    # 计算边的权重值
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # 3. 创建稀疏张量
    p_matrix = torch.sparse_coo_tensor(edge_index, edge_weight,
                                        torch.Size([num_nodes, num_nodes]),
                                        dtype=dtype,
                                        device=device)

    return p_matrix

