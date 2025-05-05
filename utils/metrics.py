# utils/metrics.py

import numpy as np
from sklearn.metrics import roc_auc_score as sk_roc_auc_score
from sklearn.metrics import average_precision_score as sk_average_precision_score

def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    计算 ROC AUC (Area Under the Receiver Operating Characteristic Curve) 分数。

    Args:
        y_true (np.ndarray): 真实的二分类标签 (0 或 1)。形状为 (num_samples,)。
        y_score (np.ndarray): 模型预测的分数或概率 (通常是 logits 或 sigmoid 输出)。
                              值越大表示越可能为正类 (1)。形状为 (num_samples,)。

    Returns:
        float: ROC AUC 分数。
    """
    # 检查 y_true 中是否同时包含正负样本，否则 AUC 无定义
    if len(np.unique(y_true)) < 2:
        print("警告: roc_auc_score - 真实标签只包含一个类别，AUC 未定义，返回 0.5")
        return 0.5
    try:
        return float(sk_roc_auc_score(y_true, y_score))
    except ValueError as e:
        # 处理一些 sklearn 可能抛出的错误，例如输入为空
        print(f"计算 ROC AUC 时出错: {e}")
        return 0.0 # 或者返回 NaN 或其他指示错误的值

def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    计算 Average Precision (AP) 分数，也称为 PR AUC (Area Under the Precision-Recall Curve)。

    Args:
        y_true (np.ndarray): 真实的二分类标签 (0 或 1)。形状为 (num_samples,)。
        y_score (np.ndarray): 模型预测的分数或概率。形状为 (num_samples,)。

    Returns:
        float: Average Precision 分数。
    """
    # 检查 y_true 中是否有正样本，否则 AP 无定义
    if np.sum(y_true) == 0:
        print("警告: average_precision_score - 真实标签中没有正样本，AP 未定义，返回 0.0")
        return 0.0
    try:
        return float(sk_average_precision_score(y_true, y_score))
    except ValueError as e:
        print(f"计算 Average Precision 时出错: {e}")
        return 0.0

# --- 示例用法 ---
if __name__ == '__main__':
    print("测试 metrics.py...")

    # 示例数据
    y_true_np = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    # 模拟模型输出的 logits 或概率
    y_score_np = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.7, 0.3, 0.9])

    print("\n示例数据:")
    print("y_true:", y_true_np)
    print("y_score:", y_score_np)

    # 计算 AUC
    auc = roc_auc_score(y_true_np, y_score_np)
    print(f"\nROC AUC Score: {auc:.4f}")

    # 计算 AP
    ap = average_precision_score(y_true_np, y_score_np)
    print(f"Average Precision Score: {ap:.4f}")

    # 测试边界情况
    print("\n测试边界情况:")
    y_true_only_zeros = np.array([0, 0, 0])
    y_score_dummy = np.array([0.1, 0.2, 0.3])
    auc_zeros = roc_auc_score(y_true_only_zeros, y_score_dummy)
    ap_zeros = average_precision_score(y_true_only_zeros, y_score_dummy)
    print(f"  只有负样本: AUC={auc_zeros}, AP={ap_zeros}")

    y_true_only_ones = np.array([1, 1, 1])
    auc_ones = roc_auc_score(y_true_only_ones, y_score_dummy)
    ap_ones = average_precision_score(y_true_only_ones, y_score_dummy)
    print(f"  只有正样本: AUC={auc_ones}, AP={ap_ones}")