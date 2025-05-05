# models/dynamic_freq_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

# 导入我们定义的模块
from .filters import LowPassFilterLayer, HighPassFilterLayer # 或者导入 BaseFilterLayer 然后实例化
from .sequence_encoder import LSTMWrapper

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


class DynamicFrequencyGNN(nn.Module):
    def __init__(self,
                 num_node_features: int,
                 lpf_out_dim: int,   
                 hpf_out_dim: int,   
                 lstm_hidden_dim: int,
                 lstm_layers: int = 1,
                 lstm_dropout: float = 0.0,
                 link_pred_hidden_dim: Optional[int] = None, 
                 use_activation_in_filters: bool = True):    
        super().__init__()

        self.lpf_out_dim = lpf_out_dim
        self.hpf_out_dim = hpf_out_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        self.low_pass_filter = LowPassFilterLayer(
            num_node_features, lpf_out_dim, activation=use_activation_in_filters
        )
        self.high_pass_filter = HighPassFilterLayer(
            num_node_features, hpf_out_dim, activation=use_activation_in_filters
        )

        lstm_input_dim = lpf_out_dim + hpf_out_dim

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
        if num_snapshots == 0:
            raise ValueError("输入快照数据列表不能为空")

        num_nodes = snapshots_data[0]['features'].shape[0]
        device = snapshots_data[0]['features'].device

        lstm_inputs = [] 
        for t in range(num_snapshots):
            features_t = snapshots_data[t]['features']
            p_matrix_t = snapshots_data[t]['p_matrix']

            low_freq_repr_t = self.low_pass_filter(features_t, p_matrix_t) 
            high_freq_repr_t = self.high_pass_filter(features_t, p_matrix_t) 

            combined_repr_t = torch.cat([low_freq_repr_t, high_freq_repr_t], dim=1) 
            lstm_inputs.append(combined_repr_t)

        lstm_input_tensor_seq_first = torch.stack(lstm_inputs, dim=0)
        lstm_input_tensor = lstm_input_tensor_seq_first.permute(1, 0, 2)

        lstm_output_sequence, (final_h, final_c) = self.lstm_encoder(lstm_input_tensor)
        final_node_repr = lstm_output_sequence[:, -1, :] 
        src_nodes_repr = final_node_repr[predict_edge_index[0]] # [num_predict_edges, lstm_hidden_dim]
        dst_nodes_repr = final_node_repr[predict_edge_index[1]] # [num_predict_edges, lstm_hidden_dim]

        link_logits = self.link_predictor(src_nodes_repr, dst_nodes_repr) # [num_predict_edges, 1]

        return link_logits.squeeze(-1) # 返回 [num_predict_edges] 形状的 logits

