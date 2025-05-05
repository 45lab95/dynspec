# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
import os
from typing import List, Dict, Tuple
from tqdm import tqdm

# 导入项目模块
from utils.data_loader import load_bitcoin_otc_data, get_dynamic_data_splits, generate_node_features
from utils.metrics import roc_auc_score, average_precision_score # 假设 metrics.py 中有这两个函数
from utils.torch_utils import set_seed, get_device # 假设 torch_utils.py 中有这些函数
from models.model import DynamicFrequencyGNN # 导入主模型

def train_one_epoch(model: DynamicFrequencyGNN,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    train_snapshots: List[Dict[str, torch.Tensor]], # 输入快照 0..T_train-1
                    train_target_edges: List[Dict[str, torch.Tensor]], # 目标边 t=1..T_train
                    device: torch.device) -> float:
    """执行一个训练轮次 (预测新增链接)"""
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    num_processed_steps = 0

    # 迭代训练时间步，用 0..t-1 的输入预测 t 的新增链接
    for t in range(len(train_snapshots)): # t 从 0 到 T_train-1
        # 模型输入是到当前步 t 的所有快照
        model_input_snapshots = []
        for i in range(t + 1): # 输入序列长度从 1 增长到 T_train
            snapshot_data = {
                'features': train_snapshots[i]['features'].to(device),
                'p_matrix': train_snapshots[i]['p_matrix'].to(device)
            }
            model_input_snapshots.append(snapshot_data)

        # 目标是 t+1 时刻的新增链接
        target_edges_t_plus_1 = train_target_edges[t] # train_target_edges 的索引 t 对应原始时间步 t+1
        target_pos_edges = target_edges_t_plus_1['pos_edge_index'].to(device)
        target_neg_edges = target_edges_t_plus_1['neg_edge_index'].to(device)

        # 如果当前目标时间步没有新增边或负样本，可以跳过
        if target_pos_edges.numel() == 0 or target_neg_edges.numel() == 0:
            continue

        # 准备预测的边索引
        predict_edge_index = torch.cat([target_pos_edges, target_neg_edges], dim=1)

        # 模型前向传播
        # 模型需要能处理可变长度的输入序列
        logits = model(model_input_snapshots, predict_edge_index)

        # 准备目标标签
        pos_labels = torch.ones(target_pos_edges.size(1), device=device)
        neg_labels = torch.zeros(target_neg_edges.size(1), device=device)
        labels = torch.cat([pos_labels, neg_labels])

        # 计算损失 (可以累加或者每个 step 都 backward)
        loss = criterion(logits, labels)
        # 这里简化为每个 step 都反向传播，也可以累积
        loss.backward() # 计算梯度
        total_loss += loss.item()
        num_processed_steps += 1

    # 在所有时间步处理完后（或分批处理完后）执行优化
    if num_processed_steps > 0:
        optimizer.step() # 更新参数
        optimizer.zero_grad() # 清空梯度以备下一轮
        avg_loss = total_loss / num_processed_steps
    else:
        avg_loss = 0.0


    return avg_loss # 返回平均损失


@torch.no_grad()
def evaluate(model: DynamicFrequencyGNN,
             input_snapshots: List[Dict[str, torch.Tensor]], # 输入快照 0..T_eval_start-1
             target_edges_list: List[Dict[str, torch.Tensor]], # 目标边 T_eval_start .. T_eval_end
             device: torch.device) -> Tuple[float, float]:
    """在验证集或测试集上评估模型 (预测新增链接，计算平均指标)"""
    model.eval()

    all_logits = []
    all_labels = []

    # 准备输入快照 (只需要准备一次)
    model_input_snapshots = []
    for i in range(len(input_snapshots)):
         snapshot_data = {
            'features': input_snapshots[i]['features'].to(device),
            'p_matrix': input_snapshots[i]['p_matrix'].to(device)
        }
         model_input_snapshots.append(snapshot_data)

    # 迭代评估时间段内的每个目标时间步
    for t in range(len(target_edges_list)):
        target_edges_t = target_edges_list[t]
        target_pos_edges = target_edges_t['pos_edge_index'].to(device)
        target_neg_edges = target_edges_t['neg_edge_index'].to(device)

        # 如果当前目标时间步没有新增边或负样本，跳过
        if target_pos_edges.numel() == 0 or target_neg_edges.numel() == 0:
            continue

        # 准备预测的边索引
        predict_edge_index = torch.cat([target_pos_edges, target_neg_edges], dim=1)

        # 模型前向传播 (使用固定的历史输入)
        logits = model(model_input_snapshots, predict_edge_index)

        # 准备目标标签
        pos_labels = torch.ones(target_pos_edges.size(1), device=device)
        neg_labels = torch.zeros(target_neg_edges.size(1), device=device)
        labels = torch.cat([pos_labels, neg_labels])

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    # 如果没有任何可评估的时间步，返回 0
    if not all_logits:
        print("警告: 评估期间没有找到有效的正负样本对。")
        return 0.0, 0.0

    # 将所有时间步的 logits 和 labels 合并
    final_logits = torch.cat(all_logits).numpy()
    final_labels = torch.cat(all_labels).numpy()

    # 计算总体 AUC 和 AP
    auc = roc_auc_score(final_labels, final_logits)
    ap = average_precision_score(final_labels, final_logits)

    # 打印评估期间的信息 (移除了之前的单步打印)
    print(f"\n--- Evaluating ---")
    print(f"  评估时间步范围: {len(target_edges_list)} 个")
    print(f"  总评估正样本边数量: {np.sum(final_labels)}")
    print(f"  总评估负样本边数量: {len(final_labels) - np.sum(final_labels)}")
    print(f"--------------------")

    return auc, ap


def main():
    # --- 参数设置 ---
    parser = argparse.ArgumentParser(description='Dynamic Frequency GNN Training')
    # 数据集参数
    parser.add_argument('--dataset_root', type=str, default='./bitcoin_otc_pyg_raw', help='PyG 数据集根目录')
    parser.add_argument('--edge_window_size', type=int, default=10, help='BitcoinOTC 边窗口大小')
    # 模型超参数
    parser.add_argument('--lpf_dim', type=int, default=64, help='低通滤波器输出维度')
    parser.add_argument('--hpf_dim', type=int, default=64, help='高通滤波器输出维度')
    parser.add_argument('--lstm_hidden', type=int, default=128, help='LSTM 隐藏层维度')
    parser.add_argument('--lstm_layers', type=int, default=1, help='LSTM 层数')
    parser.add_argument('--lstm_dropout', type=float, default=0.0, help='LSTM 层间 dropout')
    parser.add_argument('--link_pred_hidden', type=int, default=64, help='链接预测头隐藏维度 (None 表示无隐藏层)')
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减')
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集时间步比例')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集时间步比例')
    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='auto', help='计算设备 (auto, cpu, cuda:0)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--model_name', type=str, default='dyn_spectral_bitcoin.pt', help='模型文件名')

    args = parser.parse_args()
    print("--- 配置参数 ---")
    print(args)

    # --- 设置环境 ---
    set_seed(args.seed)
    device = get_device(args.device) # 自动选择 GPU 或 CPU
    os.makedirs(args.save_dir, exist_ok=True)
    model_save_path = os.path.join(args.save_dir, args.model_name)

    # --- 加载和准备数据 ---
    print("\n--- 加载数据 ---")
    snapshots_data, num_nodes, feature_dim = load_bitcoin_otc_data(
        root=args.dataset_root,
        edge_window_size=args.edge_window_size,
        feature_generator=generate_node_features
    )
    num_time_steps = len(snapshots_data)
    train_steps_idx, val_steps_idx, test_steps_idx = get_dynamic_data_splits(
        num_time_steps, args.train_ratio, args.val_ratio
    )

    # --- 准备训练、验证、测试所需的数据子集 (调整目标边索引) ---
    # 训练输入：快照 0 到 T_train-1
    train_snapshots = [snapshots_data[i] for i in train_steps_idx]
    # 训练目标：新增边 t=1 到 T_train ( snapshots_data[1] 到 snapshots_data[T_train] 的 pos/neg)
    train_target_edges = [
        {'pos_edge_index': snapshots_data[i]['pos_edge_index'],
         'neg_edge_index': snapshots_data[i]['neg_edge_index']}
        for i in range(1, train_steps_idx[-1] + 2) # 索引从 1 开始到 T_train
    ] # 长度为 T_train

    # 验证输入：快照 0 到 T_val_start-1 (即 0 到 T_train-1)
    val_input_snapshots = [snapshots_data[i] for i in train_steps_idx]
    # 验证目标：新增边 T_val_start 到 T_val_end (即 T_train 到 T_train+T_val-1)
    val_target_edges = [
        {'pos_edge_index': snapshots_data[i]['pos_edge_index'],
         'neg_edge_index': snapshots_data[i]['neg_edge_index']}
        for i in val_steps_idx # 使用验证集的时间步索引 (T_train 到 T_train+T_val-1)
    ] # 长度为 T_val

    # 测试输入：快照 0 到 T_test_start-1 (即 0 到 T_train+T_val-1)
    test_input_snapshots = [snapshots_data[i] for i in range(test_steps_idx[0])]
    # 测试目标：新增边 T_test_start 到 T_test_end (即 T_train+T_val 到 T-1)
    test_target_edges = [
        {'pos_edge_index': snapshots_data[i]['pos_edge_index'],
         'neg_edge_index': snapshots_data[i]['neg_edge_index']}
        for i in test_steps_idx # 使用测试集的时间步索引
    ] # 长度为 T_test

    print(f"\n数据准备完成: Train input steps={len(train_snapshots)}, Train target steps={len(train_target_edges)}")
    print(f"Val input steps={len(val_input_snapshots)}, Val target steps={len(val_target_edges)}")
    print(f"Test input steps={len(test_input_snapshots)}, Test target steps={len(test_target_edges)}")

    # --- 初始化模型、优化器、损失函数 ---
    print("\n--- 初始化模型 ---")
    model = DynamicFrequencyGNN(
        num_node_features=feature_dim,
        lpf_out_dim=args.lpf_dim,
        hpf_out_dim=args.hpf_dim,
        lstm_hidden_dim=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        link_pred_hidden_dim=args.link_pred_hidden
    ).to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss() # 适用于 logits 输出的二分类交叉熵

    # --- 训练循环 ---
    print("\n--- 开始训练 ---")
    best_val_auc = 0.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # 训练 (传入调整后的目标列表)
        loss = train_one_epoch(model, optimizer, criterion, train_snapshots,
                               train_target_edges, device) # 传入目标边列表

        # 验证 (传入调整后的目标列表)
        val_auc, val_ap = evaluate(model, val_input_snapshots, val_target_edges, device) # 传入目标边列表

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.2f}s | Avg Loss: {loss:.4f} | Val Avg AUC: {val_auc:.4f} | Val Avg AP: {val_ap:.4f}")

        # 早停和模型保存
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            print(f"  New best validation AUC found! Saving model to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Validation AUC did not improve for {args.patience} epochs. Early stopping.")
                break

    total_train_time = time.time() - start_time
    print(f"\n--- 训练完成 --- Total Time: {total_train_time:.2f}s")

    # --- 测试 ---
    print("\n--- 开始测试 ---")
    print(f"Loading best model from {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))
    test_auc, test_ap = evaluate(model, test_input_snapshots, test_target_edges, device)
    print(f"Test Results --> Avg AUC: {test_auc:.4f} | Avg AP: {test_ap:.4f}")

if __name__ == "__main__":
    main()