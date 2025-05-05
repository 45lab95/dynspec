# utils/preprocess_utils.py

import torch
from torch_geometric.utils import add_self_loops, degree
from torch_sparse import coalesce 

def compute_propagation_matrix(edge_index: torch.Tensor,
                               num_nodes: int,
                               add_self_loops_flag: bool = True,
                               dtype: torch.dtype = torch.float) -> torch.Tensor:
    """
    计算对称归一化的邻接矩阵 (传播矩阵 P)，通常用于 GCN 等模型。
    P = D^{-1/2} * A_tilde * D^{-1/2}，其中 A_tilde = A + I。

    Args:
        edge_index (torch.Tensor): 边列表，形状为 [2, num_edges]。
        num_nodes (int): 图中的节点数量。
        add_self_loops_flag (bool, optional): 是否在计算前添加自环。默认为 True。
        dtype (torch.dtype, optional): 返回的稀疏张量的数据类型。默认为 torch.float。

    Returns:
        torch.Tensor: 对称归一化的邻接矩阵 (传播矩阵 P)，
                      以 PyTorch 稀疏张量 (COO 格式) 表示。
    """
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

# --- 示例用法 (可以放在 if __name__ == "__main__": 中测试) ---
if __name__ == '__main__':
    edge_idx = torch.tensor([[0, 1, 1, 2],
                             [1, 0, 2, 1]], dtype=torch.long)
    num_nodes = 4

    print("原始 Edge Index:\n", edge_idx)

    # 计算传播矩阵 (添加自环)
    p_matrix = compute_propagation_matrix(edge_idx, num_nodes)
    print("\n计算得到的传播矩阵 P (稀疏 COO 格式):\n", p_matrix)
    # 可以转换为稠密矩阵查看 (仅适用于小图)
    print("\n转换为稠密矩阵:\n", p_matrix.to_dense())
