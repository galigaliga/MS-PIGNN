import numpy as np
import torch
import random
import os
import pandas as pd

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Metrics:
    @staticmethod
    def MSE(pred, true):
        return np.mean((pred - true) ** 2)

    @staticmethod
    def MAE(pred, true):
        return np.mean(np.abs(pred - true))

    @staticmethod
    def RMSE(pred, true):
        return np.sqrt(np.mean((pred - true) ** 2))

    @staticmethod
    def R2(pred, true):
        # R2 = 1 - (SS_res / SS_tot)
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        # 避免分母为0
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def MASE(pred, true):
        # 计算分子: 模型的 MAE
        mae_model = np.mean(np.abs(pred - true))
        true_flat = true.flatten()
        n = len(true_flat)
        if n < 2: return 1.0  # 避免除零
        d = np.abs(np.diff(true_flat)).mean()

        if d == 0: return 0.0
        return mae_model / d