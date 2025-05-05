import torch
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.utils import to_undirected
import numpy as np
from tqdm import tqdm # 用于显示进度条 (pip install tqdm)

def edge_index_to_set(edge_index):
    """将 edge_index 转换为边的集合，忽略方向并去重"""
    # 确保是无向图表示中最小的在前，方便比较
    edges = edge_index.cpu().numpy()
    # 对每条边排序，例如 (1, 0) -> (0, 1)
    sorted_edges = np.sort(edges, axis=0)
    # 转换为元组集合以去重
    return set(map(tuple, sorted_edges.T))

def check_bitcoin_otc_dynamics(root: str = './bitcoin_otc_pyg_raw', edge_window_size: int = 10):
    """
    检查 BitcoinOTC 数据集时间片之间的边变化情况。
    """
    print(f"加载 BitcoinOTC 数据集 (edge_window_size={edge_window_size})...")
    try:
        dataset = BitcoinOTC(root=root, edge_window_size=edge_window_size)
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return

    if not dataset:
        print("数据集为空!")
        return

    num_nodes = dataset[0].num_nodes
    print(f"数据集加载成功，节点数: {num_nodes}，时间步数: {len(dataset)}")

    previous_edge_set = set()
    total_added = 0
    total_removed = 0
    total_kept = 0
    total_current = 0

    print("\n计算边变化统计 (相对于上一时间步):")
    for t, data_t in enumerate(tqdm(dataset, desc="处理时间步")):
        # 获取当前边，并确保无向化
        current_edge_index = to_undirected(data_t.edge_index, num_nodes=num_nodes)
        current_edge_set = edge_index_to_set(current_edge_index)

        if t > 0: # 从第二个时间步开始比较
            added_edges = current_edge_set - previous_edge_set
            removed_edges = previous_edge_set - current_edge_set
            kept_edges = current_edge_set.intersection(previous_edge_set)

            num_added = len(added_edges)
            num_removed = len(removed_edges)
            num_kept = len(kept_edges)
            num_current = len(current_edge_set)

            print(f"时间步 {t}: 新增边 = {num_added}, 消失边 = {num_removed}, "
                  f"保持边 = {num_kept}, 当前总边数 = {num_current}")

            total_added += num_added
            total_removed += num_removed
            total_kept += num_kept # Kept 会重复计算，只关注增删
            total_current += num_current
        else:
            # 第一个时间步没有比较对象
            num_current = len(current_edge_set)
            print(f"时间步 {t}: 当前总边数 = {num_current} (初始状态)")
            total_current += num_current


        previous_edge_set = current_edge_set

    print("\n--- 汇总统计 ---")
    avg_added = total_added / (len(dataset) - 1) if len(dataset) > 1 else 0
    avg_removed = total_removed / (len(dataset) - 1) if len(dataset) > 1 else 0
    avg_current = total_current / len(dataset) if len(dataset) > 0 else 0

    print(f"平均每步新增边数: {avg_added:.2f}")
    print(f"平均每步消失边数: {avg_removed:.2f}")
    print(f"平均每步总边数: {avg_current:.2f}")
    if avg_current > 0:
        avg_change_rate = (avg_added + avg_removed) / avg_current * 100
        print(f"平均变化率 ((新增+消失)/总数): {avg_change_rate:.2f}%")


if __name__ == "__main__":
    # 你可以在这里修改 edge_window_size 来观察其影响
    check_bitcoin_otc_dynamics(edge_window_size=10)
    # check_bitcoin_otc_dynamics(edge_window_size=1) # 比较窗口为 1 的情况