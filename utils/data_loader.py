
import torch
import numpy as np
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.utils import degree, to_undirected, negative_sampling
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from .preprocess_utils import compute_propagation_matrix
from .unibasis_utils import compute_unibasis_for_snapshot

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
                          feature_generator = generate_node_features,
                          K: int = 10, # 新增：UniBasis 阶数
                          tau: float = 0.5, # 新增：UniBasis 混合系数
                          h_hat_global: Optional[float] = 0.5 # 新增：全局估计同配率 (可选)
                          ) -> Tuple[List[Dict[str, torch.Tensor]], int, int]:
    """
    加载并预处理 BitcoinOTC 数据集，计算 UniBasis 特征，并为预测新增链接准备标签。

    Args:
        root (str): PyG 数据集存储的根目录。
        edge_window_size (int): 边的持续窗口大小。
        feature_generator (callable): 用于生成节点特征的函数。
        K (int): UniBasis 的最大阶数。
        tau (float): UniBasis 的混合系数 τ。
        h_hat_global (Optional[float]): 全局使用的估计同配率 ĥ。如果为 None，
                                        则需要实现动态计算 h_hat_t 的逻辑 (此处简化)。
                                        默认为 0.5。

    Returns:
        Tuple[List[Dict[str, torch.Tensor]], int, int]:
            - snapshots_data: 包含每个时间步数据的列表，每个元素是字典：
                {
                    'p_matrix': 传播矩阵 P_t [N, N] (sparse),
                    'unibasis_features': 计算得到的 UniBasis 特征 [N, (K+1)*F],
                    'pos_edge_index': 当前时间步新增的正样本边 [2, num_pos],
                    'neg_edge_index': 当前时间步的负采样边 [2, num_neg]
                }
            - num_nodes: 图中的节点数量。
            - feature_dim_F: 单个基向量的特征维度 F。
    """
    print(f"加载 BitcoinOTC 数据集 (edge_window_size={edge_window_size})...")
    dataset = BitcoinOTC(root=root, edge_window_size=edge_window_size)
    num_nodes = dataset[0].num_nodes
    feature_dim_F = -1 # 初始化单个基向量的特征维度 F
    snapshots_data = []
    previous_edge_set = set()

    # --- 处理 h_hat ---
    # 在这个简化版本中，我们直接使用全局 h_hat
    # 更复杂的版本可以在这里基于历史数据动态计算 h_hat_t
    if h_hat_global is None:
        print("警告: 未提供 h_hat_global，将使用默认值 0.5。建议提供或实现动态计算。")
        h_hat_current = 0.5
    else:
        h_hat_current = h_hat_global
        print(f"使用全局估计同配率 h_hat = {h_hat_current:.4f}")


    print("开始处理时间步并计算 UniBasis...")
    for t, data_t in enumerate(tqdm(dataset, desc="处理时间步")):
        current_edge_index_raw = data_t.edge_index
        device = current_edge_index_raw.device

        current_edge_index = to_undirected(current_edge_index_raw, num_nodes=num_nodes)
        current_edge_set = edge_index_to_set(current_edge_index)

        # 1. 计算传播矩阵 P_t
        p_matrix_t = compute_propagation_matrix(current_edge_index, num_nodes)

        # 2. 生成 *初始* 节点特征 X_t (用于 UniBasis 计算)
        initial_features_t = feature_generator(current_edge_index, num_nodes)
        if feature_dim_F == -1:
            feature_dim_F = initial_features_t.shape[1] # 获取特征维度 F

        # 3. 计算 UniBasis 特征
        # 调用我们之前创建的函数
        unibasis_features_t, _ = compute_unibasis_for_snapshot(
            p_matrix=p_matrix_t,
            features=initial_features_t,
            K=K,
            tau=tau,
            h_hat=h_hat_current # 使用当前的 h_hat
        )
        # unibasis_features_t 的形状是 [N, (K+1)*F]

        # 4. 准备链接预测标签 (新增链接)
        if t > 0:
            new_pos_edges_set = current_edge_set - previous_edge_set
            pos_edge_index_t = set_to_edge_index(new_pos_edges_set, device)

            num_pos_samples = pos_edge_index_t.size(1) // 2
            num_neg_samples = num_pos_samples if num_pos_samples > 0 else 1000 # 与之前逻辑相同

            neg_edge_index_t = negative_sampling(
                edge_index=current_edge_index,
                num_nodes=num_nodes,
                num_neg_samples=num_neg_samples,
                method='sparse'
            )
        else:
            pos_edge_index_t = torch.empty((2, 0), dtype=torch.long, device=device)
            neg_edge_index_t = torch.empty((2, 0), dtype=torch.long, device=device)

        # 存储处理好的数据
        snapshots_data.append({
            'p_matrix': p_matrix_t, # 保留 P_t，可能未来需要
            'unibasis_features': unibasis_features_t, # 存储 UniBasis 特征
            'pos_edge_index': pos_edge_index_t,
            'neg_edge_index': neg_edge_index_t
        })

        previous_edge_set = current_edge_set

    print(f"\n数据处理完成 (标签为新增链接, 特征为 UniBasis [N, (K+1)*F]). 单个基维度 F={feature_dim_F}")
    # 返回包含 UniBasis 特征的列表、节点数、单个基的维度 F
    return snapshots_data, num_nodes, feature_dim_F

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

