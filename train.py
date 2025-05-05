# train.py (修改版)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
import os
from typing import List, Dict, Tuple
from tqdm import tqdm # 如果在 data_loader 中使用了 tqdm，这里可能不需要

# 导入项目模块
# 确保能正确导入修改后的 data_loader
from utils.data_loader import load_bitcoin_otc_data, get_dynamic_data_splits, generate_node_features
from utils.metrics import roc_auc_score, average_precision_score
from utils.torch_utils import set_seed, get_device
# 确保能正确导入修改后的模型
from models.model import DynamicFrequencyGNN, Combination # 可能需要导入 Combination 用于优化器参数分组
# 导入绘图工具
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.plot_utils import plot_training_curves

# --- train_one_epoch 函数 (逻辑不变，但参数传递给 model 的部分改变) ---
def train_one_epoch(model: DynamicFrequencyGNN,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    train_snapshots: List[Dict[str, torch.Tensor]],
                    train_target_edges: List[Dict[str, torch.Tensor]],
                    device: torch.device) -> float:
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    num_processed_steps = 0

    # 迭代训练时间步
    for t in tqdm(range(len(train_snapshots)), desc="Training Steps", leave=False): # 可选添加进度条
        model_input_snapshots = []
        for i in range(t + 1):
            snapshot_data = {
                # 主要变化：传递 'unibasis_features'
                'unibasis_features': train_snapshots[i]['unibasis_features'].to(device),
                # 'p_matrix' 仍然存在于 snapshots_data 中，但模型 forward 不再直接使用它
            }
            model_input_snapshots.append(snapshot_data)

        target_edges_t_plus_1 = train_target_edges[t]
        target_pos_edges = target_edges_t_plus_1['pos_edge_index'].to(device)
        target_neg_edges = target_edges_t_plus_1['neg_edge_index'].to(device)

        if target_pos_edges.numel() == 0 or target_neg_edges.numel() == 0: continue

        predict_edge_index = torch.cat([target_pos_edges, target_neg_edges], dim=1)

        # 模型前向传播 (现在使用包含 UniBasis 的输入)
        logits = model(model_input_snapshots, predict_edge_index)

        pos_labels = torch.ones(target_pos_edges.size(1), device=device)
        neg_labels = torch.zeros(target_neg_edges.size(1), device=device)
        labels = torch.cat([pos_labels, neg_labels])

        loss = criterion(logits, labels)
        loss.backward()
        total_loss += loss.item()
        num_processed_steps += 1

    if num_processed_steps > 0:
        optimizer.step()
        optimizer.zero_grad()
        avg_loss = total_loss / num_processed_steps
    else:
        avg_loss = 0.0

    return avg_loss

# --- evaluate 函数 (逻辑不变，但参数传递给 model 的部分改变) ---
@torch.no_grad()
def evaluate(model: DynamicFrequencyGNN,
             input_snapshots: List[Dict[str, torch.Tensor]],
             target_edges_list: List[Dict[str, torch.Tensor]],
             device: torch.device) -> Tuple[float, float]:
    model.eval()
    all_logits = []
    all_labels = []

    # 准备模型输入 (一次性)
    model_input_snapshots = []
    for i in range(len(input_snapshots)):
         snapshot_data = {
            # 主要变化：传递 'unibasis_features'
            'unibasis_features': input_snapshots[i]['unibasis_features'].to(device),
         }
         model_input_snapshots.append(snapshot_data)

    # 迭代评估目标时间步
    for t in tqdm(range(len(target_edges_list)), desc="Evaluating Steps", leave=False): # 可选添加进度条
        target_edges_t = target_edges_list[t]
        target_pos_edges = target_edges_t['pos_edge_index'].to(device)
        target_neg_edges = target_edges_t['neg_edge_index'].to(device)

        if target_pos_edges.numel() == 0 or target_neg_edges.numel() == 0: continue

        predict_edge_index = torch.cat([target_pos_edges, target_neg_edges], dim=1)

        # 模型前向传播 (使用包含 UniBasis 的输入)
        logits = model(model_input_snapshots, predict_edge_index)

        pos_labels = torch.ones(target_pos_edges.size(1), device=device)
        neg_labels = torch.zeros(target_neg_edges.size(1), device=device)
        labels = torch.cat([pos_labels, neg_labels])

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    if not all_logits:
        print("警告: 评估期间没有找到有效的正负样本对。")
        return 0.0, 0.0

    final_logits = torch.cat(all_logits).numpy()
    final_labels = torch.cat(all_labels).numpy()

    auc = roc_auc_score(final_labels, final_logits)
    ap = average_precision_score(final_labels, final_logits)

    # 保持评估信息打印
    print(f"\n--- Evaluating ---")
    print(f"  评估时间步范围: {len(target_edges_list)} 个")
    print(f"  总评估正样本边数量: {np.sum(final_labels)}")
    print(f"  总评估负样本边数量: {len(final_labels) - np.sum(final_labels)}")
    print(f"--------------------")


    return auc, ap


def main():
    # --- 参数设置 (进行修改) ---
    parser = argparse.ArgumentParser(description='Dynamic Graph Learning with UniBasis')
    # 数据集参数
    parser.add_argument('--dataset_root', type=str, default='./bitcoin_otc_pyg_raw', help='PyG 数据集根目录')
    parser.add_argument('--edge_window_size', type=int, default=10, help='BitcoinOTC 边窗口大小')
    # UniBasis 参数
    parser.add_argument('--K', type=int, default=10, help='UniBasis 多项式阶数')
    parser.add_argument('--tau', type=float, default=0.5, help='UniBasis 同配/异配混合系数 τ')
    parser.add_argument('--h_hat_global', type=float, default=0.5, help='全局估计同配率 ĥ (用于 BitcoinOTC)')
    # 模型超参数
    parser.add_argument('--combination_dropout', type=float, default=0.3, help='Combination 层 dropout') # 参考 UniFilter dpC
    parser.add_argument('--lstm_hidden', type=int, default=128, help='LSTM 隐藏层维度')
    parser.add_argument('--lstm_layers', type=int, default=1, help='LSTM 层数')
    parser.add_argument('--lstm_dropout', type=float, default=0.0, help='LSTM 层间 dropout')
    parser.add_argument('--link_pred_hidden', type=int, default=64, help='链接预测头隐藏维度')
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr_comb', type=float, default=0.005, help='Combination 层学习率') # 参考 UniFilter lr2
    parser.add_argument('--lr_other', type=float, default=0.01, help='模型其他部分学习率 (LSTM, LinkPred)')
    parser.add_argument('--wd_comb', type=float, default=0.0, help='Combination 层权重衰减') # 参考 UniFilter wd2
    parser.add_argument('--wd_other', type=float, default=0.0, help='模型其他部分权重衰减')
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集时间步比例')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集时间步比例')
    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='auto', help='计算设备 (auto, cpu, cuda:0)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--model_name', type=str, default='dyn_unibasis_bitcoin.pt', help='模型文件名') # 更新模型名

    args = parser.parse_args()
    print("--- 配置参数 ---")
    print(args)

    # --- 设置环境 ---
    set_seed(args.seed)
    device = get_device(args.device)
    plot_save_dir = "plots"
    plot_filename = f"curves_{args.model_name.replace('.pt', '')}.png"
    os.makedirs(args.save_dir, exist_ok=True)
    model_save_path = os.path.join(args.save_dir, args.model_name)

    # --- 加载和准备数据 (传入 UniBasis 参数) ---
    print("\n--- 加载数据 ---")
    snapshots_data, num_nodes, feature_dim_F = load_bitcoin_otc_data(
        root=args.dataset_root,
        edge_window_size=args.edge_window_size,
        feature_generator=generate_node_features,
        K=args.K, # 传入 K
        tau=args.tau, # 传入 tau
        h_hat_global=args.h_hat_global # 传入 h_hat
    )
    num_time_steps = len(snapshots_data)
    train_steps_idx, val_steps_idx, test_steps_idx = get_dynamic_data_splits(
        num_time_steps, args.train_ratio, args.val_ratio
    )

    # --- 准备训练、验证、测试数据子集 (逻辑不变) ---
    # ... (与上一版 train.py 相同) ...
    train_snapshots = [snapshots_data[i] for i in train_steps_idx]
    train_target_edges = [{'pos_edge_index': snapshots_data[i+1]['pos_edge_index'], 'neg_edge_index': snapshots_data[i+1]['neg_edge_index']} for i in train_steps_idx]
    val_input_snapshots = [snapshots_data[i] for i in train_steps_idx]
    val_target_edges = [{'pos_edge_index': snapshots_data[i]['pos_edge_index'], 'neg_edge_index': snapshots_data[i]['neg_edge_index']} for i in val_steps_idx]
    test_input_snapshots = [snapshots_data[i] for i in range(test_steps_idx[0])]
    test_target_edges = [{'pos_edge_index': snapshots_data[i]['pos_edge_index'], 'neg_edge_index': snapshots_data[i]['neg_edge_index']} for i in test_steps_idx]
    # ... (打印数据准备信息) ...

    # --- 初始化模型 (传入 UniBasis 参数) ---
    print("\n--- 初始化模型 ---")
    model = DynamicFrequencyGNN(
        unibasis_base_feature_dim=feature_dim_F, # 使用 data_loader 返回的 F
        K=args.K,
        combination_dropout=args.combination_dropout, # 使用新参数
        lstm_hidden_dim=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        link_pred_hidden_dim=args.link_pred_hidden
    ).to(device)
    print(model)

    # --- 设置优化器 (使用参数组) ---
    optimizer = optim.Adam([
        {'params': model.combination.parameters(), # Combination 层的参数
         'lr': args.lr_comb, 'weight_decay': args.wd_comb},
        {'params': model.lstm_encoder.parameters(), # LSTM 层的参数
         'lr': args.lr_other, 'weight_decay': args.wd_other},
        {'params': model.link_predictor.parameters(), # 预测头的参数
         'lr': args.lr_other, 'weight_decay': args.wd_other}
    ])
    print("\n优化器已设置参数组。")

    criterion = nn.BCEWithLogitsLoss()

    # --- 初始化训练历史记录 (不变) ---
    training_history = { 'epoch': [], 'train_loss': [], 'val_auc': [], 'val_ap': [] }

    # --- 训练循环 (逻辑不变) ---
    print("\n--- 开始训练 ---")
    best_val_auc = 0.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        # ... (调用 train_one_epoch, evaluate, 记录 history, 早停逻辑不变) ...
        epoch_start_time = time.time()
        loss = train_one_epoch(model, optimizer, criterion, train_snapshots, train_target_edges, device)
        val_auc, val_ap = evaluate(model, val_input_snapshots, val_target_edges, device)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.2f}s | Avg Loss: {loss:.4f} | Val Avg AUC: {val_auc:.4f} | Val Avg AP: {val_ap:.4f}")
        training_history['epoch'].append(epoch + 1); training_history['train_loss'].append(loss); training_history['val_auc'].append(val_auc); training_history['val_ap'].append(val_ap)
        if val_auc > best_val_auc: best_val_auc = val_auc; patience_counter = 0; print(f"  New best validation AUC found! Saving model to {model_save_path}"); torch.save(model.state_dict(), model_save_path)
        else: patience_counter += 1;
        if patience_counter >= args.patience: print(f"  Early stopping."); break


    total_train_time = time.time() - start_time
    print(f"\n--- 训练完成 --- Total Time: {total_train_time:.2f}s")

    # --- 绘制曲线 (逻辑不变) ---
    try:
        plot_training_curves(training_history, title=f"Training Curves ({args.model_name.replace('.pt', '')})", save_dir=plot_save_dir, filename=plot_filename)
    except Exception as e: print(f"绘制训练曲线时发生错误: {e}")

    # --- 测试 (逻辑不变) ---
    print("\n--- 开始测试 ---")
    if os.path.exists(model_save_path): # 检查最佳模型文件是否存在
        print(f"Loading best model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))
        test_auc, test_ap = evaluate(model, test_input_snapshots, test_target_edges, device)
        print(f"Test Results --> Avg AUC: {test_auc:.4f} | Avg AP: {test_ap:.4f}")
    else:
        print("错误：找不到保存的最佳模型，无法进行测试。")


if __name__ == "__main__":
    main()