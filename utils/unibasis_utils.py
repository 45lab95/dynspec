import torch
import torch.nn.functional as F # 虽然下面没直接用 F，但依赖的库可能需要
import math
import gc # UniFilter 代码中使用了 gc
from typing import Tuple

def compute_unibasis_for_snapshot(p_matrix: torch.Tensor,
                                  features: torch.Tensor,
                                  K: int,
                                  tau: float,
                                  h_hat: float) -> Tuple[torch.Tensor, int]:
    """
    为单个时间步计算 UniBasis 基向量 (移植自 UniFilter)。

    Args:
        p_matrix (torch.Tensor): 当前时间步的传播矩阵 P_t [N, N] (sparse)。
                                 通常是对称归一化邻接矩阵 A_sym。
        features (torch.Tensor): 当前时间步的初始节点特征 X_t [N, F]。
        K (int): 多项式的最大阶数。
        tau (float): 同配/异配基的混合系数 τ。
        h_hat (float): 估计的同配率 ĥ，用于计算目标角度。

    Returns:
        Tuple[torch.Tensor, int]:
            - unibasis_features (torch.Tensor): 拼接后的 UniBasis 特征矩阵
                                                 [N, (K+1)*F]。
            - feature_dim (int): 单个基向量的特征维度 F。
    """
    num_nodes, feature_dim = features.shape
    device = features.device
    cosval = math.cos(math.pi * (1.0 - h_hat) / 2.0)

    norm_feat = torch.norm(features, dim=0, keepdim=True) 
    norm_feat = torch.clamp(norm_feat, min=1e-8)
    u_0 = features / norm_feat 

    v_last = u_0 
    v_second_last = torch.zeros_like(v_last, device=device) 
    basis_sum = torch.zeros_like(u_0, device=device)
    basis_sum += u_0 
    hm_k = features # HM_0 = X_t

   
    unibasis_list = [hm_k * tau + u_0 * (1.0 - tau)] 


    for k in range(1, K + 1):

        v_k_temp = torch.spmm(p_matrix, v_last)
        project_1 = torch.einsum('nd,nd->d', v_k_temp, v_last)
        project_2 = torch.einsum('nd,nd->d', v_k_temp, v_second_last)
        v_k_orth = v_k_temp - project_1 * v_last - project_2 * v_second_last
        norm_vk = torch.norm(v_k_orth, dim=0, keepdim=True)
        norm_vk = torch.clamp(norm_vk, min=1e-8)
        v_k = v_k_orth / norm_vk
        hm_k = torch.spmm(p_matrix, hm_k)
        H_k_approx = basis_sum / k
        last_unibasis = unibasis_list[-1]

        term1_numerator = torch.einsum('nd,nd->d', H_k_approx, last_unibasis)
        term1_sq = torch.square(term1_numerator / cosval) if cosval != 0 else torch.zeros_like(term1_numerator) # 避免除零

        term2 = ((k - 1) * cosval + 1) / k
        Tf_sq = torch.clamp(term1_sq - term2, min=0.0)
        Tf = torch.sqrt(Tf_sq)
        u_k_unnormalized = H_k_approx + torch.mul(Tf, v_k)
        norm_uk = torch.norm(u_k_unnormalized, dim=0, keepdim=True)
        norm_uk = torch.clamp(norm_uk, min=1e-8)
        u_k = u_k_unnormalized / norm_uk 
        norm_hmk = torch.norm(hm_k, dim=0, keepdim=True)
        norm_hmk = torch.clamp(norm_hmk, min=1e-8)
        hm_k_normalized = hm_k / norm_hmk

        b_k = hm_k_normalized * tau + u_k * (1.0 - tau)
        unibasis_list.append(b_k)
        basis_sum += u_k 
        v_second_last = v_last 
        v_last = v_k      


    del v_last, v_second_last, basis_sum, hm_k 
    gc.collect()
    unibasis_features = torch.cat(unibasis_list, dim=1)

    return unibasis_features, feature_dim
