
import torch
import numpy as np
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.utils import degree, to_undirected, negative_sampling
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from .preprocess_utils import compute_propagation_matrix

def generate_node_features(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    return deg.view(-1, 1)

def edge_index_to_set(edge_index):
    """将 edge_index 转换为无向边的集合，方便进行集合运算"""
    if edge_index.numel() == 0:
        return set()
    edges = edge_index.cpu().numpy()
    # 确保无向表示唯一 (u, v) 其中 u < v
    edges_sorted = np.sort(edges, axis=0)
    return set(map(tuple, edges_sorted.T))

def set_to_edge_index(edge_set: set, device: torch.device) -> torch.Tensor:
    """将边的集合转换回 edge_index Tensor"""
    if not edge_set:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    edges = torch.tensor(list(edge_set), dtype=torch.long, device=device).t()
    # 添加反向边以确保无向
    return to_undirected(edges)



def load_bitcoin_otc_data(root: str = './bitcoin_otc_pyg_raw',
                          edge_window_size: int = 10,
                          feature_generator = generate_node_features
                          ) -> Tuple[List[Dict[str, torch.Tensor]], int, int]:
    """
    加载并预处理 BitcoinOTC 数据集，为 *预测新增链接* 准备标签。
    """
    print(f"加载 BitcoinOTC 数据集 (edge_window_size={edge_window_size})...")
    dataset = BitcoinOTC(root=root, edge_window_size=edge_window_size)
    num_nodes = dataset[0].num_nodes
    feature_dim = -1
    snapshots_data = []
    previous_edge_set = set() # 用于存储上一个时间步的边集合

    print("开始处理时间步 (为预测新增链接准备标签)...")
    for t, data_t in enumerate(tqdm(dataset, desc="处理时间步")):
        current_edge_index_raw = data_t.edge_index
        device = current_edge_index_raw.device # 获取设备

        # 确保边是无向的
        current_edge_index = to_undirected(current_edge_index_raw, num_nodes=num_nodes)
        current_edge_set = edge_index_to_set(current_edge_index)

        # 1. 计算传播矩阵 P_t (基于当前所有边)
        p_matrix_t = compute_propagation_matrix(current_edge_index, num_nodes)

        # 2. 生成节点特征 X_t (基于当前所有边)
        features_t = feature_generator(current_edge_index, num_nodes)
        if feature_dim == -1:
            feature_dim = features_t.shape[1]

        # 3. 准备链接预测标签 (用于 t-1 预测 t 的 *新增* 链接)
        if t > 0: # 从第二个时间步开始，才能计算新增链接
            # 正样本：当前存在但上一步不存在的边 E_t \ E_{t-1}
            new_pos_edges_set = current_edge_set - previous_edge_set
            # 将集合转回 edge_index 格式 (确保无向)
            pos_edge_index_t = set_to_edge_index(new_pos_edges_set, device)

            # 负样本：当前仍然不存在的边
            # 需要采样不在 current_edge_set 中的边
            # 注意：负采样数量可以与正样本数量一致，或者更多
            num_pos_samples = pos_edge_index_t.size(1) // 2 # 除以2因为 set_to_edge_index 加了反向边
            if num_pos_samples == 0: # 如果没有新增边，也采一些负样本（或者跳过这个时间步的损失计算）
                num_neg_samples = 1000 # 或者其他合理值
            else:
                num_neg_samples = num_pos_samples # 与新增正样本数量一致

            # 进行负采样，确保不采样到 current_edge_set 中的边
            # PyG 的 negative_sampling 函数第一个参数是 *包含* 要避免的边的 edge_index
            neg_edge_index_t = negative_sampling(
                edge_index=current_edge_index, # 传入当前所有存在的边，避免采样到它们
                num_nodes=num_nodes,
                num_neg_samples=num_neg_samples,
                method='sparse'
            )
        else: # 第一个时间步没有前一步，无法定义新增链接，标签设为空
            pos_edge_index_t = torch.empty((2, 0), dtype=torch.long, device=device)
            neg_edge_index_t = torch.empty((2, 0), dtype=torch.long, device=device)


        snapshots_data.append({
            'features': features_t,
            'p_matrix': p_matrix_t,
            'pos_edge_index': pos_edge_index_t, # 存储的是新增的边
            'neg_edge_index': neg_edge_index_t  # 存储的是当前不存在的边
        })

        # 更新上一步的边集合
        previous_edge_set = current_edge_set

    print("\n数据处理完成 (标签为新增链接).")
    # 注意：返回的数据现在第一个时间步的 pos/neg edge_index 是空的
    return snapshots_data, num_nodes, feature_dim

def get_dynamic_data_splits(num_time_steps: int,
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15
                           ) -> Tuple[List[int], List[int], List[int]]:
    
    T = num_time_steps
    T_train = int(T * train_ratio)
    T_val = int(T * val_ratio)
    T_test = T - T_train - T_val

    if T_train <= 0 or T_val <= 0 or T_test <= 0:
        raise ValueError("划分比例导致某个集合的时间步数量小于等于 0")

    # 时间步索引从 0 开始
    train_steps = list(range(T_train))
    val_steps = list(range(T_train, T_train + T_val))
    test_steps = list(range(T_train + T_val, T))

    print(f"时间步划分: Train={len(train_steps)} ({train_steps[0]}-{train_steps[-1]}), "
          f"Val={len(val_steps)} ({val_steps[0]}-{val_steps[-1]}), "
          f"Test={len(test_steps)} ({test_steps[0]}-{test_steps[-1]})")

    return train_steps, val_steps, test_steps

