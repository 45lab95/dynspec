# models/dynamic_freq_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any


# 导入项目模块
from .base_model import BaseDynamicBackbone, BaseTaskHead # 导入基类
from .dyn_spectral import DynSpectralBackbone    # 导入我们实现的 Backbone
from .task_head import LinkPredictorHead                # 导入链接预测头
from .sequence_encoder import LSTMWrapper # DynSpectralBackbone 会用到

class DynSpectral(nn.Module):

    def __init__(self,
                 device: torch.device,
                 unibasis_base_feature_dim: int,
                 K: int,
                 combination_dropout: float,
                 lstm_hidden_dim: int,
                 lstm_layers: int = 1,
                 lstm_dropout: float = 0.0,
                 link_pred_hidden_dim: Optional[int] = None,
                 ):

        super().__init__()
        self.device = device

        self.backbone: BaseDynamicBackbone = DynSpectralBackbone(
            device=device,
            unibasis_base_feature_dim=unibasis_base_feature_dim,
            K=K,
            combination_dropout=combination_dropout,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            lstm_dropout=lstm_dropout
        ).to(device)


        self.task_head: BaseTaskHead = LinkPredictorHead(
            node_embedding_dim=lstm_hidden_dim, # 来自 Backbone 的输出
            hidden_dim=link_pred_hidden_dim
        ).to(device)

        # (可选) 调用参数初始化
        self.reset_parameters()

    def forward(self,
                snapshots_data: List[Dict[str, torch.Tensor]],
                target_edges: Optional[torch.Tensor] = None, # 对于链接预测是必需的
                **kwargs: Any
                ) -> torch.Tensor:

       
        node_representations = self.backbone(snapshots_data, **kwargs)

        if isinstance(self.task_head, LinkPredictorHead):
            if target_edges is None:
                raise ValueError("LinkPredictorHead 需要 target_edges 参数。")
            logits = self.task_head(node_representations, target_edges, **kwargs)
        else:
            logits = self.task_head(node_representations, target_edges=target_edges, **kwargs)

        return logits

    def reset_parameters(self):
        """(可选) 重置整个模型的参数"""
        print("DynSpectral (Wrapper): 重置参数...")
        if hasattr(self.backbone, 'reset_parameters'):
            self.backbone.reset_parameters()
        if hasattr(self.task_head, 'reset_parameters'):
            self.task_head.reset_parameters()
