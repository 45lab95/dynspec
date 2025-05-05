
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from .filters import LowPassFilterLayer, HighPassFilterLayer 
from .sequence_encoder import LSTMWrapper
import numpy as np
import math 



class Combination(nn.Module):
    '''
    A mod combination the bases of polynomial filters.
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

        if total_unibasis_dim % (self.K + 1) != 0:
            raise ValueError(f"UniBasis 特征维度 ({total_unibasis_dim}) "
                             f"无法被 K+1 ({self.K+1}) 整除。")
        single_feature_dim_F_calculated = total_unibasis_dim // (self.K + 1)
        lstm_inputs = []


        for t in range(num_snapshots):

            unibasis_features_t = snapshots_data[t]['unibasis_features']

            try:
                unibasis_t_reshaped = unibasis_features_t.view(
                    num_nodes, self.K + 1, current_F # 使用 current_F
                )
            except RuntimeError as e:
                 print(f"在时间步 {t} reshape UniBasis 特征时出错: {e}")
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
