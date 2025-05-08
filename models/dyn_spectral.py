import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any

# 导入基类和我们需要的模块
from .base_model import BaseDynamicBackbone
from .filters import Combination # 从我们修改后的 model.py 导入 Combination
from .sequence_encoder import LSTMWrapper

class DynSpectralBackbone(BaseDynamicBackbone):
    """
    动态谱图主干网络 (使用 UniBasis 和 LSTM)。
    这个类负责从动态图快照中提取时序节点表示。
    """
    def __init__(self,
                 device: torch.device,
                 unibasis_base_feature_dim: int, # 单个 UniBasis 基向量的维度 F
                 K: int,                         # UniBasis 的阶数
                 combination_dropout: float,
                 lstm_hidden_dim: int,
                 lstm_layers: int = 1,
                 lstm_dropout: float = 0.0):
        """
        初始化 DynSpectralBackbone。

        Args:
            device (torch.device): 模型运行的设备。
            unibasis_base_feature_dim (int): 单个 UniBasis 基向量的特征维度 F。
            K (int): UniBasis 的最大阶数。
            combination_dropout (float): Combination 层的 dropout 概率。
            lstm_hidden_dim (int): LSTM 隐藏状态的维度。
            lstm_layers (int, optional): LSTM 的层数。默认为 1。
            lstm_dropout (float, optional): LSTM 层间的 dropout 概率。默认为 0.0。
        """
        super().__init__(device) # 调用父类初始化

        self.unibasis_base_feature_dim = unibasis_base_feature_dim
        self.K = K
        self.lstm_hidden_dim = lstm_hidden_dim

        # 1. 实例化 Combination 层
        self.combination = Combination(
            channels=unibasis_base_feature_dim,
            level=K + 1, # level 是基的数量 K+1
            dropout=combination_dropout
        ).to(device) # 移动到指定设备

        # 2. 实例化 LSTM 包装器
        # LSTM 的输入维度是 Combination 层的输出维度，即 F
        lstm_input_dim = unibasis_base_feature_dim
        self.lstm_encoder = LSTMWrapper(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True, # 确保 batch_first=True
            dropout=lstm_dropout
        ).to(device) # 移动到指定设备

        # (可选) 调用参数初始化
        self.reset_parameters() # 如果需要特定的初始化

    def forward(self,
                snapshots_data: List[Dict[str, torch.Tensor]],
                **kwargs: Any
                ) -> torch.Tensor: # 返回最后一个时间步的节点表示
        """
        主干网络的前向传播。

        Args:
            snapshots_data (List[Dict[str, torch.Tensor]]):
                包含 T 个时间步图快照数据的列表。每个字典需要包含:
                {'unibasis_features': UniBasis 特征 [N, (K+1)*F]}
                (N 是节点数, K 是阶数, F 是单个基维度)
            **kwargs: 未使用的额外参数，保持接口一致性。

        Returns:
            torch.Tensor: 最后一个时间步的节点表示 [N, lstm_hidden_dim]。
        """
        num_snapshots = len(snapshots_data)
        if num_snapshots == 0:
            raise ValueError("输入快照列表不能为空")

        # 获取节点数量 N (假设固定) 和单个基特征维度 F
        num_nodes = snapshots_data[0]['unibasis_features'].shape[0]
        # (K+1)*F
        total_unibasis_dim = snapshots_data[0]['unibasis_features'].shape[1]

        # 计算 F (并进行验证)
        if total_unibasis_dim % (self.K + 1) != 0:
            raise ValueError(f"UniBasis 特征维度 ({total_unibasis_dim}) "
                             f"无法被 K+1 ({self.K+1}) 整除。")
        calculated_F = total_unibasis_dim // (self.K + 1)
        if calculated_F != self.unibasis_base_feature_dim:
            print(f"警告 (DynSpectralBackbone): 计算得到的单个基维度 F ({calculated_F}) "
                  f"与模型初始化时的维度 ({self.unibasis_base_feature_dim}) 不符。将使用初始化维度。")
        current_F = self.unibasis_base_feature_dim # 优先使用初始化时传入的 F

        lstm_inputs_for_sequence = []

        # --- 迭代时间步，应用 Combination 层 ---
        for t in range(num_snapshots):
            unibasis_features_t = snapshots_data[t]['unibasis_features'].to(self.device)

            # Reshape 为 Combination 层期望的格式 [N, K+1, F]
            try:
                unibasis_t_reshaped = unibasis_features_t.view(
                    num_nodes, self.K + 1, current_F
                )
            except RuntimeError as e:
                print(f"DynSpectralBackbone: 在时间步 {t} reshape UniBasis 特征时出错: {e}")
                print(f"  原始形状: {unibasis_features_t.shape}")
                print(f"  目标形状: ({num_nodes}, {self.K + 1}, {current_F})")
                raise e

            # 应用 Combination 层进行加权求和
            combined_repr_t = self.combination(unibasis_t_reshaped) # [N, F]
            lstm_inputs_for_sequence.append(combined_repr_t)

        # --- 准备 LSTM 输入并进行时序编码 ---
        if not lstm_inputs_for_sequence: # 如果输入序列为空 (例如 num_snapshots=0 经过了上面检查后, 或者循环内部有continue)
            # 需要返回一个合理形状的零张量或者抛出错误
            # 这里假设至少有一个时间步，否则上面的 num_snapshots 检查会捕获
             raise ValueError("LSTM 输入序列为空，无法继续。")


        # 堆叠: [T, N, F]
        lstm_input_tensor_seq_first = torch.stack(lstm_inputs_for_sequence, dim=0)
        # 调整为 batch_first: [N, T, F] (N 是节点数，视为 batch_size)
        lstm_input_tensor_batch_first = lstm_input_tensor_seq_first.permute(1, 0, 2)

        # LSTM 编码
        # 不提供初始状态，LSTMWrapper 内部的 nn.LSTM 会自动使用零状态
        lstm_output_sequence, (final_h, final_c) = self.lstm_encoder(lstm_input_tensor_batch_first)
        # lstm_output_sequence 形状: [N, T, lstm_hidden_dim]

        # 返回最后一个时间步 T 的 LSTM 输出作为最终的节点表示
        final_node_representations = lstm_output_sequence[:, -1, :] # [N, lstm_hidden_dim]

        return final_node_representations



    def reset_parameters(self):
        print("DynSpectralBackbone: 重置参数...")
        if hasattr(self.combination, 'reset_parameters'):
            self.combination.reset_parameters()
        pass 