"""
评估指标模块

对应论文 "Tracking Performance" 节（Table 2）：
  - MAE   (Mean Absolute Error)：平均绝对误差
  - RMSE  (Root Mean Squared Error)：均方根误差
  - 50th/90th 百分位误差：中位数误差和 90 百分位误差

定位误差均以"米"为单位，衡量估计位置与真实位置的欧氏距离。
"""

import numpy as np
import torch
from typing import Union, Tuple


def position_error(pred: Union[np.ndarray, torch.Tensor],
                   gt: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    计算每个时间步的欧氏位置误差（单位：米）。

    Args:
        pred: [..., 2]  — 预测位置 [x, y]
        gt:   [..., 2]  — 真实位置 [x, y]

    Returns:
        errors: [...]  — 每个样本的欧氏误差（米）
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    diff = pred - gt
    return np.sqrt((diff ** 2).sum(axis=-1))  # 欧氏距离


def compute_mae(pred: Union[np.ndarray, torch.Tensor],
                gt: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算平均绝对误差（MAE）。
    对应 Table 2 第一列。

    MAE = (1/N) Σ_i ||p̂_i - p_i||

    Returns:
        float  — 平均位置误差（米）
    """
    errors = position_error(pred, gt)
    return float(np.mean(errors))


def compute_rmse(pred: Union[np.ndarray, torch.Tensor],
                 gt: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算均方根误差（RMSE）。
    对应 Table 2 第二列。

    RMSE = sqrt( (1/N) Σ_i ||p̂_i - p_i||² )

    Note: 先算欧氏距离再平均取根（非对各坐标分量分别计算）

    Returns:
        float  — RMSE（米）
    """
    errors = position_error(pred, gt)
    return float(np.sqrt(np.mean(errors ** 2)))


def compute_percentile_error(pred: Union[np.ndarray, torch.Tensor],
                              gt: Union[np.ndarray, torch.Tensor],
                              percentile: float = 50.0) -> float:
    """
    计算给定百分位的误差（CDF 误差）。
    对应 Table 2 第三/四列（50th, 90th）。

    Args:
        percentile: 百分位数（0-100），常用 50（中位数）和 90

    Returns:
        float  — 对应百分位误差（米）
    """
    errors = position_error(pred, gt)
    return float(np.percentile(errors, percentile))


def compute_all_metrics(pred: Union[np.ndarray, torch.Tensor],
                        gt: Union[np.ndarray, torch.Tensor]
                        ) -> dict:
    """
    一次性计算所有评估指标，对应 Table 2 的一行。

    Returns:
        dict 包含：
          'MAE':  平均绝对误差（米）
          'RMSE': 均方根误差（米）
          '50th': 50 百分位误差（米）
          '90th': 90 百分位误差（米）
    """
    errors = position_error(pred, gt)
    return {
        'MAE':  float(np.mean(errors)),
        'RMSE': float(np.sqrt(np.mean(errors ** 2))),
        '50th': float(np.percentile(errors, 50)),
        '90th': float(np.percentile(errors, 90)),
    }


def print_metrics(metrics: dict, method_name: str = ""):
    """
    格式化打印评估结果（对应 Table 2 的格式）。

    示例输出：
      LocDreamer-scheduling | MAE: 0.66 | RMSE: 0.77 | 50th: 0.60 | 90th: 1.18
    """
    prefix = f"{method_name:<35}" if method_name else ""
    print(
        f"{prefix} | "
        f"MAE: {metrics['MAE']:.3f} | "
        f"RMSE: {metrics['RMSE']:.3f} | "
        f"50th: {metrics['50th']:.3f} | "
        f"90th: {metrics['90th']:.3f}"
    )
