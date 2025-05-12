# utils/data_loader.py

import torch
import numpy as np
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.utils import degree, to_undirected, negative_sampling
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import math
import gc
import pandas as pd # 用于方便地处理时间戳和分组
import datetime # 用于处理日期


from .preprocess_utils import compute_propagation_matrix

def edge_index_to_set(edge_index: torch.Tensor) -> set: # 添加类型提示
    """将 edge_index 转换为无向边的集合，方便进行集合运算"""
    if edge_index.numel() == 0:
        return set()
    # 确保 edge_index 在 CPU 上进行 NumPy 操作
    edges = edge_index.cpu().numpy()
    # 确保无向表示唯一 (u, v) 其中 u < v
    edges_sorted = np.sort(edges, axis=0)
    return set(map(tuple, edges_sorted.T))

def set_to_edge_index(edge_set: set, device: torch.device) -> torch.Tensor:
    """将边的集合转换回 edge_index Tensor (确保输出是无向的)"""
    if not edge_set:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    # 从集合创建边列表
    edges_list = list(edge_set)
    if not edges_list: # 再次检查，因为 list(empty_set) 是 []
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edges = torch.tensor(edges_list, dtype=torch.long, device=device).t() # [2, num_unique_undirected_edges]

    # 因为我们的集合存储的是 u < v 的唯一无向边，
    # to_undirected 会为每条边 (u,v) 添加 (v,u) 并移除重复。
    # 如果原始集合已经是排序好的无向边，直接添加反向边然后合并可能更直接。
    # 但 to_undirected 更通用和安全。
    return to_undirected(edges)


def generate_node_features(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    生成节点特征。这里使用节点度作为特征。

    Args:
        edge_index (torch.Tensor): 当前时间步的边列表 [2, num_edges]。
        num_nodes (int): 节点数量。

    Returns:
        torch.Tensor: 节点特征矩阵 [num_nodes, 1]。
    """
    # 计算度数 (确保使用 float 类型以便后续计算)
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    # 将度数作为一维特征返回
    return deg.view(-1, 1)


def compute_unibasis_for_snapshot(p_matrix: torch.Tensor,
                                  features: torch.Tensor,
                                  K: int,
                                  tau: float,
                                  h_hat: float) -> Tuple[torch.Tensor, int]:

    num_nodes, feature_dim = features.shape
    device = features.device
    cosval = math.cos(math.pi * (1.0 - h_hat) / 2.0)

    norm_feat = torch.norm(features, dim=0, keepdim=True) 
    norm_feat = torch.clamp(norm_feat, min=1e-8)
    u_0 = features / norm_feat 

    v_last = u_0 
    v_second_last = torch.zeros_like(v_last, device=device) 
    basis_sum = torch.zeros_like(u_0, device=device)
    basis_sum += u_0 
    hm_k = features # HM_0 = X_t

   
    unibasis_list = [hm_k * tau + u_0 * (1.0 - tau)] 


    for k in range(1, K + 1):

        v_k_temp = torch.spmm(p_matrix, v_last)
        project_1 = torch.einsum('nd,nd->d', v_k_temp, v_last)
        project_2 = torch.einsum('nd,nd->d', v_k_temp, v_second_last)
        v_k_orth = v_k_temp - project_1 * v_last - project_2 * v_second_last
        norm_vk = torch.norm(v_k_orth, dim=0, keepdim=True)
        norm_vk = torch.clamp(norm_vk, min=1e-8)
        v_k = v_k_orth / norm_vk
        hm_k = torch.spmm(p_matrix, hm_k)
        H_k_approx = basis_sum / k
        last_unibasis = unibasis_list[-1]

        term1_numerator = torch.einsum('nd,nd->d', H_k_approx, last_unibasis)
        term1_sq = torch.square(term1_numerator / cosval) if cosval != 0 else torch.zeros_like(term1_numerator) # 避免除零

        term2 = ((k - 1) * cosval + 1) / k
        Tf_sq = torch.clamp(term1_sq - term2, min=0.0)
        Tf = torch.sqrt(Tf_sq)
        u_k_unnormalized = H_k_approx + torch.mul(Tf, v_k)
        norm_uk = torch.norm(u_k_unnormalized, dim=0, keepdim=True)
        norm_uk = torch.clamp(norm_uk, min=1e-8)
        u_k = u_k_unnormalized / norm_uk 
        norm_hmk = torch.norm(hm_k, dim=0, keepdim=True)
        norm_hmk = torch.clamp(norm_hmk, min=1e-8)
        hm_k_normalized = hm_k / norm_hmk

        b_k = hm_k_normalized * tau + u_k * (1.0 - tau)
        unibasis_list.append(b_k)
        basis_sum += u_k 
        v_second_last = v_last 
        v_last = v_k      


    del v_last, v_second_last, basis_sum, hm_k 
    gc.collect()
    unibasis_features = torch.cat(unibasis_list, dim=1)

    return unibasis_features, feature_dim


def load_uc_irvine_message_data(
    data_path: str, # 原始数据文件路径, e.g., "data/uc_irvine_messages/out.opsahl-ucsocial"
    feature_generator=generate_node_features,
    K: int = 10,
    tau: float = 0.5,
    h_hat_global: Optional[float] = 0.5,
    min_nodes_remap: bool = True # 是否将节点ID从0开始重新映射
) -> Tuple[List[Dict[str, torch.Tensor]], int, int]:

    print(f"加载 UC Irvine messages 数据集从: {data_path}")
    print(f"参数: K={K}, tau={tau}, h_hat={h_hat_global}")

    # 1. 读取原始数据文件
    # 假设文件是空白分隔，并且至少有4列: src, dst, weight, timestamp
    try:
        # 使用 pandas 读取可以方便处理不同数量的列和时间转换
        df = pd.read_csv(data_path, sep=r'\s+', header=None, comment='%',
                         names=['src', 'dst', 'weight', 'timestamp'])
    except pd.errors.ParserError:
        print("尝试解析为3列 (无权重)...")
        try:
            df = pd.read_csv(data_path, sep=r'\s+', header=None, comment='%',
                            names=['src', 'dst', 'timestamp'])
            df['weight'] = 1 # 如果没有权重列，默认权重为1
        except Exception as e:
            raise ValueError(f"无法解析数据文件 {data_path}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"数据文件未找到: {data_path}")

    print(f"  原始数据共 {len(df)} 条边/事件。")

    if min_nodes_remap:
        all_nodes = pd.unique(df[['src', 'dst']].values.ravel('K'))
        node_map = {node_id: i for i, node_id in enumerate(all_nodes)}
        df['src'] = df['src'].map(node_map)
        df['dst'] = df['dst'].map(node_map)
        num_nodes = len(all_nodes)
        print(f"  节点 ID 已重新映射，总节点数: {num_nodes}")
    else:
        # 假设原始节点ID已经是0到N-1，或者用户自己处理
        num_nodes = max(df['src'].max(), df['dst'].max()) + 1
        print(f"  使用原始节点 ID，推断总节点数: {num_nodes} (确保ID从0开始)")


    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    df = df.sort_values(by='timestamp') # 按时间排序
    unique_dates = sorted(df['date'].unique())
    print(f"  数据覆盖从 {unique_dates[0]} 到 {unique_dates[-1]}，共 {len(unique_dates)} 天。")
    snapshots_intermediate = []
    feature_dim_F = -1 # 初始化

    if h_hat_global is None:
        h_hat_current = 0.5
    else:
        h_hat_current = h_hat_global
    print(f"  使用全局估计同配率 h_hat = {h_hat_current:.4f}")

    print("步骤 1: 按天构建图快照并计算 UniBasis...")
    for current_date in tqdm(unique_dates, desc="构建每日快照"):
        # 筛选当前日期的边
        edges_for_day = df[df['date'] == current_date][['src', 'dst']].values.astype(np.int64)

        if edges_for_day.shape[0] == 0: # 当天没有边
            edge_index_t = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index_t = torch.from_numpy(edges_for_day.T)

        edge_index_t_undirected = to_undirected(edge_index_t, num_nodes=num_nodes)
        device = edge_index_t_undirected.device 

 
        p_matrix_t = compute_propagation_matrix(edge_index_t_undirected, num_nodes)

        initial_features_t = feature_generator(edge_index_t_undirected, num_nodes)
        if feature_dim_F == -1:
            feature_dim_F = initial_features_t.shape[1]

        unibasis_features_t, _ = compute_unibasis_for_snapshot(
            p_matrix=p_matrix_t,
            features=initial_features_t,
            K=K, tau=tau, h_hat=h_hat_current
        )

        snapshots_intermediate.append({
            'unibasis_features': unibasis_features_t,
            'current_edges': edge_index_t_undirected # 存储当天的无向边，用于下一个快照的标签
        })

    snapshots_data_final = []
    print("\n步骤 2: 准备链接预测标签...")
    for t in tqdm(range(len(snapshots_intermediate) - 1), desc="准备标签"):
        current_day_info = snapshots_intermediate[t]
        next_day_info = snapshots_intermediate[t+1]
        device = current_day_info['unibasis_features'].device

        pos_edge_index_label = next_day_info['current_edges']

        neg_edge_index_label = negative_sampling(
            edge_index=pos_edge_index_label,
            num_nodes=num_nodes,
            num_neg_samples=pos_edge_index_label.size(1) if pos_edge_index_label.numel() > 0 else 1000, # 保证有负样本
            method='sparse'
        )

        snapshots_data_final.append({
            'unibasis_features': current_day_info['unibasis_features'],
            'pos_edge_index': pos_edge_index_label,
            'neg_edge_index': neg_edge_index_label
        })

    print(f"\nUC Irvine 数据处理完成。共生成 {len(snapshots_data_final)} 个 (特征, 标签对) 快照。")
    print(f"  节点数量: {num_nodes}")
    print(f"  单个 UniBasis 基向量特征维度 F: {feature_dim_F}")
    return snapshots_data_final, num_nodes, feature_dim_F


def load_bitcoin_otc_data(root: str = './bitcoin_otc_pyg_raw',
                          edge_window_size: int = 10,
                          feature_generator = generate_node_features,
                          K: int = 10,
                          tau: float = 0.5,
                          h_hat_global: Optional[float] = 0.5
                          ) -> Tuple[List[Dict[str, torch.Tensor]], int, int]:
    print(f"加载 BitcoinOTC 数据集 (edge_window_size={edge_window_size}, K={K}, tau={tau}, h_hat={h_hat_global})...")
    dataset = BitcoinOTC(root=root, edge_window_size=edge_window_size)
    num_nodes = dataset[0].num_nodes
    feature_dim_F = -1
    snapshots_data_intermediate = [] # 先存储每个时间步的 X_t 和 P_t

    if h_hat_global is None:
        h_hat_current = 0.5
    else:
        h_hat_current = h_hat_global
    print(f"使用全局估计同配率 h_hat = {h_hat_current:.4f}")

    print("步骤 1: 处理每个时间步的特征和传播矩阵...")
    for t, data_t in enumerate(tqdm(dataset, desc="处理快照")):
        current_edge_index_raw = data_t.edge_index
        # device = current_edge_index_raw.device # 在下面获取

        current_edge_index = to_undirected(current_edge_index_raw, num_nodes=num_nodes)
        device = current_edge_index.device # 从处理后的 edge_index 获取 device

        p_matrix_t = compute_propagation_matrix(current_edge_index, num_nodes)
        initial_features_t = feature_generator(current_edge_index, num_nodes)
        if feature_dim_F == -1:
            feature_dim_F = initial_features_t.shape[1]

        unibasis_features_t, _ = compute_unibasis_for_snapshot(
            p_matrix=p_matrix_t,
            features=initial_features_t,
            K=K, tau=tau, h_hat=h_hat_current
        )

        snapshots_data_intermediate.append({
            'unibasis_features': unibasis_features_t,
            'current_edges': current_edge_index # 存储当前时间步的真实边 (用于下一个时间步的标签)
        })

    # --- 步骤 2: 准备链接预测标签 (t 时刻的输入预测 t+1 时刻的链接) ---
    snapshots_data_final = []
    print("\n步骤 2: 准备链接预测标签...")
    # 迭代到倒数第二个快照，因为最后一个快照没有“下一个”时间步来作为标签
    for t in tqdm(range(len(snapshots_data_intermediate) - 1), desc="准备标签"):
        current_snapshot_info = snapshots_data_intermediate[t]
        next_snapshot_info = snapshots_data_intermediate[t+1]
        device = current_snapshot_info['unibasis_features'].device # 获取设备

        # 正样本：下一个时间步 (t+1) 实际存在的边
        pos_edge_index_label = next_snapshot_info['current_edges']

        # 负样本：在下一个时间步 (t+1) 不存在的边
        # 确保采样时避免采样到 t+1 时刻已存在的边
        neg_edge_index_label = negative_sampling(
            edge_index=pos_edge_index_label, # 避免采样到这些正样本
            num_nodes=num_nodes,
            num_neg_samples=pos_edge_index_label.size(1), # 与正样本数量一致
            method='sparse'
        )

        snapshots_data_final.append({
            'unibasis_features': current_snapshot_info['unibasis_features'], # t 时刻的特征
            # 'p_matrix': current_snapshot_info['p_matrix'], # 如果模型不再需要 P，可以不传
            'pos_edge_index': pos_edge_index_label, # t+1 时刻的边
            'neg_edge_index': neg_edge_index_label  # t+1 时刻的边
        })

    print(f"\n数据处理完成 (标签为预测下一时间片存在链接). 单个基维度 F={feature_dim_F}")
    # 注意：最终的 snapshots_data_final 会比原始数据集少一个时间步
    return snapshots_data_final, num_nodes, feature_dim_F


def get_dynamic_data_splits(num_time_steps: int,
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15
                           ) -> Tuple[List[int], List[int], List[int]]:
    """
    划分时间步用于训练、验证和测试。
    注意：这里是按时间步划分，模型将使用前 T_train 步的数据预测 T_train+1 步的链接。

    Args:
        num_time_steps (int): 总的时间步数量。
        train_ratio (float, optional): 训练集时间步比例。默认为 0.7。
        val_ratio (float, optional): 验证集时间步比例。默认为 0.15。

    Returns:
        Tuple[List[int], List[int], List[int]]: 训练、验证、测试的时间步索引列表。
    """
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

