# models/dynamic_freq_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

# 导入我们定义的模块
from .filters import LowPassFilterLayer, HighPassFilterLayer # 或者导入 BaseFilterLayer 然后实例化
from .sequence_encoder import LSTMWrapper
import numpy as np # Combination 类需要 numpy
import math # 可能需要 math



class Combination(nn.Module):
    '''
    A mod combination the bases of polynomial filters.
    Args:
        channels (int): number of feature channels.
        level (int): number of bases to combine.
        sole (bool): whether or not use the same filter for all output channels.
    '''
    def __init__(self, channels, level, dropout, sole=False):
        super().__init__()
        self.dropout = dropout
        self.K=level
        
        self.comb_weight = nn.Parameter(torch.ones((1, level, 1)))
        self.reset_parameters()            

    def reset_parameters(self):
        bound = 1.0/self.K
        TEMP = np.random.uniform(bound, bound, self.K)  
        self.comb_weight=nn.Parameter(torch.FloatTensor(TEMP).view(-1,self.K, 1))

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x * self.comb_weight
        x = torch.sum(x, dim=1)
        return x



class LinkPredictorHead(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.mlp = nn.Linear(input_dim * 2, 1)

    def forward(self, node_repr1: torch.Tensor, node_repr2: torch.Tensor) -> torch.Tensor:
        combined_repr = torch.cat([node_repr1, node_repr2], dim=-1)
        logit = self.mlp(combined_repr)
        return logit


# --- 修改后的 DynamicFrequencyGNN 类定义 ---
class DynamicFrequencyGNN(nn.Module):
    def __init__(self,

                 unibasis_base_feature_dim: int, 
                 K: int, 
                 combination_dropout: float,

                 lstm_hidden_dim: int,
                 lstm_layers: int = 1,
                 lstm_dropout: float = 0.0,
                 link_pred_hidden_dim: Optional[int] = None):
        super().__init__()


        self.unibasis_base_feature_dim = unibasis_base_feature_dim
        self.K = K
        self.lstm_hidden_dim = lstm_hidden_dim

        self.combination = Combination(
            channels=unibasis_base_feature_dim,
            level=K + 1,
            dropout=combination_dropout
        )


        lstm_input_dim = unibasis_base_feature_dim
        self.lstm_encoder = LSTMWrapper(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout
        )

        self.link_predictor = LinkPredictorHead(
            input_dim=lstm_hidden_dim,
            hidden_dim=link_pred_hidden_dim
        )

    def forward(self,
                snapshots_data: List[Dict[str, torch.Tensor]],
                predict_edge_index: torch.Tensor
               ) -> torch.Tensor:

        num_snapshots = len(snapshots_data)
        if num_snapshots == 0: raise ValueError("输入快照列表不能为空")


        num_nodes = snapshots_data[0]['unibasis_features'].shape[0]
        device = snapshots_data[0]['unibasis_features'].device
        total_unibasis_dim = snapshots_data[0]['unibasis_features'].shape[1]
        current_F = self.unibasis_base_feature_dim

        # 验证维度是否匹配 (可选但推荐)
        if total_unibasis_dim % (self.K + 1) != 0:
            raise ValueError(f"UniBasis 特征维度 ({total_unibasis_dim}) "
                             f"无法被 K+1 ({self.K+1}) 整除。")
        single_feature_dim_F_calculated = total_unibasis_dim // (self.K + 1)

        if single_feature_dim_F_calculated != current_F:
             print(f"警告: data_loader 计算得到的单个基维度 F ({single_feature_dim_F_calculated}) "
                   f"与模型初始化时的维度 ({current_F}) 不符。将继续使用初始化维度。")
             # 这里可以选择是否抛出错误或强制使用某个值，但默认继续使用 current_F (初始化值)

        lstm_inputs = []


        for t in range(num_snapshots):

            unibasis_features_t = snapshots_data[t]['unibasis_features']

            try:
                unibasis_t_reshaped = unibasis_features_t.view(
                    num_nodes, self.K + 1, current_F # 使用 current_F
                )
            except RuntimeError as e:
                 print(f"在时间步 {t} reshape UniBasis 特征时出错: {e}")
                 print(f"  原始形状: {unibasis_features_t.shape}")
                 print(f"  目标形状: ({num_nodes}, {self.K + 1}, {current_F})")
                 raise e

            combined_repr_t = self.combination(unibasis_t_reshaped) # [N, F]

            lstm_inputs.append(combined_repr_t)


        lstm_input_tensor_seq_first = torch.stack(lstm_inputs, dim=0)

        lstm_input_tensor = lstm_input_tensor_seq_first.permute(1, 0, 2)

        lstm_output_sequence, (final_h, final_c) = self.lstm_encoder(lstm_input_tensor)

        final_node_repr = lstm_output_sequence[:, -1, :] # [N, lstm_hidden_dim]

        src_nodes_repr = final_node_repr[predict_edge_index[0]]
        dst_nodes_repr = final_node_repr[predict_edge_index[1]]

        # 链接预测
        link_logits = self.link_predictor(src_nodes_repr, dst_nodes_repr)

        return link_logits.squeeze(-1) # 返回 [num_predict_edges]
