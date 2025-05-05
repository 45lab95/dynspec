# models/filters.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaseFilterLayer(nn.Module):
    """滤波器层的基类 """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, filter_matrix: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError("Subclasses must implement the forward method.")


class LowPassFilterLayer(BaseFilterLayer):
    """
    简单的低通滤波器层 (GCN-like)。
    执行操作: H' = σ(P * X * W)
    其中 P 是传播矩阵 (如对称归一化邻接矩阵 A_sym)。
    """
    def __init__(self, in_channels, out_channels, activation: bool = True):
        super().__init__(in_channels, out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor, p_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入节点特征 [N, in_channels]。
            p_matrix (torch.Tensor): 传播矩阵 P [N, N] (sparse)。

        Returns:
            torch.Tensor: 输出节点特征 [N, out_channels]。
        """
        # 1. 邻居聚合 (应用传播矩阵): P * X
        # torch.spmm 执行稀疏矩阵 (p_matrix) 与稠密矩阵 (x) 的乘法
        support = torch.spmm(p_matrix, x)

        # 2. 特征变换: (P * X) * W
        output = torch.mm(support, self.weight)

        # 3. (可选) 应用激活函数
        if self.activation:
            output = F.relu(output)

        return output


class HighPassFilterLayer(BaseFilterLayer):
    def __init__(self, in_channels, out_channels, activation: bool = True):
        super().__init__(in_channels, out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor, p_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入节点特征 [N, in_channels]。
            p_matrix (torch.Tensor): 传播矩阵 P [N, N] (sparse)。

        Returns:
            torch.Tensor: 输出节点特征 [N, out_channels]。
        """
        support_p = torch.spmm(p_matrix, x)

        support_i_minus_p = x - support_p

        output = torch.mm(support_i_minus_p, self.weight) 
        if self.activation:
            output = F.relu(output)

        return output
