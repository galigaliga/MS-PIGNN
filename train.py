import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import time
import pandas as pd
from utils.get_config import get_config
from utils.dataloader import get_loader
from utils.graph_utils import load_graph_data, build_input_mapping
from utils.util import setup_seed, Metrics
from test_model import MSPIGNN


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def differentiable_inverse_transform_y(tensor, mean, scale, device):
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=device, dtype=torch.float32)
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=device, dtype=torch.float32)
    return tensor * scale + mean


def differentiable_transform_y(tensor, mean, scale, device):
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=device, dtype=torch.float32)
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=device, dtype=torch.float32)
    return (tensor - mean) / (scale + 1e-8)


def main():
    args = get_config()
    setup_seed(args.seed)
    device = torch.device(args.device)


    output_base_dir = './output'
    model_save_dir = os.path.join(output_base_dir, args.model_name)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    print(f"\n==================================================")
    print(f"   PHASE 1: WARMUP TRAINING (Physical MW Loss)")
    print(f"   [Updated] Consistent with Phase 2 Strategy")
    print(f"==================================================")
    print(f"Output Directory: {model_save_dir}")
    print(f"Data Path: {args.data_path}")

    # 1. 加载数据
    print("Loading Data...")
    train_set, train_loader = get_loader(args, flag='train')
    valid_set, valid_loader = get_loader(args, flag='val')

    args.input_size = train_set.data_x.shape[1]

    # 1. 加载图拓扑
    adj_matrix, _ = load_graph_data(args.graph_path, num_nodes=118, device=device)
    args.adj = adj_matrix

    # 2. 构建物理特征映射
    input_mask = build_input_mapping(
        data_path=args.data_path,
        graph_path=args.graph_path,
        num_nodes=118,
        device=device
    )
    args.input_mask = input_mask

    # 获取反归一化参数
    y_mean = train_set.y_mean
    y_scale = train_set.y_scale

    args.y_mean = torch.tensor(y_mean, device=device).float()
    args.y_scale = torch.tensor(y_scale, device=device).float()

    # =======================================================
    # 【加载所有物理参数】(含 PTDF, Branch Limits, Gen Map)
    # =======================================================
    print(f">>> Loading Physical Parameters (Raw MW)...")

    # A. Gen Limits & Ramps
    gen_df = pd.read_csv(os.path.join(args.graph_path, 'gen.csv'))

    p_min_raw = gen_df.iloc[:, 2].values.astype(np.float32)  # Col 2
    p_max_raw = gen_df.iloc[:, 3].values.astype(np.float32)  # Col 3
    ramp_up_raw = gen_df.iloc[:, 4].values.astype(np.float32)  # Col 4
    ramp_down_raw = gen_df.iloc[:, 5].values.astype(np.float32)  # Col 5

    args.p_min = torch.tensor(p_min_raw, device=device).float()
    args.p_max = torch.tensor(p_max_raw, device=device).float()
    args.ramp_up = torch.tensor(ramp_up_raw, device=device).float()
    args.ramp_down = torch.tensor(ramp_down_raw, device=device).float()

    # B. Gen to Node Mapping (54 -> 118)
    if 'bus_id' in gen_df.columns:
        gen_node_idx = gen_df['bus_id'].values - 1
    else:
        gen_node_idx = gen_df.iloc[:, 1].values - 1
    args.gen_map_index = torch.tensor(gen_node_idx, device=device).long()

    # C. PTDF & Branch Limits
    ptdf_path = os.path.join(args.graph_path, 'ptdf.csv')
    branch_path = os.path.join(args.graph_path, 'branch.csv')

    # 读取 PTDF (假设无表头)
    if os.path.exists(ptdf_path):
        ptdf_df = pd.read_csv(ptdf_path, header=None)
        args.full_ptdf = torch.tensor(ptdf_df.values, device=device).float()
        print(f"  PTDF Shape: {args.full_ptdf.shape}")

    # 读取 Branch Limits (第8列)
    if os.path.exists(branch_path):
        branch_df = pd.read_csv(branch_path, header=0)  # 注意: 如果有表头需改 header=0
        limits_raw = branch_df.iloc[:, 7].values.astype(np.float32)
        args.full_limits = torch.tensor(limits_raw, device=device).float()
        print(f"  Branch Limits Loaded: {args.full_limits.shape}")

    # =======================================================

    # 2. 初始化模型
    print(f"Initializing {args.model_name} Model...")
    model = MSPIGNN.Model(args).to(device)
    params = count_parameters(model)
    print(f"Model Structure: {args.model_name}")
    print(f"Trainable Parameters: {params:,}")
    print(f"----------------------------------")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience)

    warmup_checkpoint = os.path.join(model_save_dir, 'warmup_checkpoint.pth')
    best_score = float('inf')

    print("\n--- Start Warmup Training ---")

    for epoch in range(args.epochs):
        t_epoch_start = time.time()

        # --- 训练 ---
        model.train()
        train_mw_losses = []  # 记录 MW Loss

        for batch_x, batch_future, batch_y, _ in train_loader:
            batch_x = batch_x.float().to(device)
            batch_future = batch_future.float().to(device)  # 【修改 2】上设备
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()

            pred_mw_final, gate_out = model(batch_x, batch_future)

            target_mw = differentiable_inverse_transform_y(batch_y, args.y_mean, args.y_scale, device)

            loss_mse_mw = criterion(pred_mw_final, target_mw)

            loss_task = torch.sqrt(loss_mse_mw + 1e-6) / 10.0

            loss_task.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_mw_losses.append(loss_task.item() * 10.0)

        # --- 验证 ---
        model.eval()
        val_preds_mw = []
        val_trues_mw = []

        with torch.no_grad():
            for batch_x, batch_future, batch_y, _ in valid_loader:
                batch_x = batch_x.float().to(device)
                batch_future = batch_future.float().to(device)  # 【修改 2】
                batch_y = batch_y.float().to(device)

                pred_mw_final, _ = model(batch_x, batch_future)

                true_mw = differentiable_inverse_transform_y(batch_y, args.y_mean, args.y_scale, device)

                val_preds_mw.append(pred_mw_final.cpu().numpy())
                val_trues_mw.append(true_mw.cpu().numpy())

        t_epoch_end = time.time()

        preds_cat = np.concatenate(val_preds_mw, axis=0)
        trues_cat = np.concatenate(val_trues_mw, axis=0)

        val_mae = Metrics.MAE(preds_cat, trues_cat)
        val_rmse_metric = Metrics.RMSE(preds_cat, trues_cat)
        val_r2 = Metrics.R2(preds_cat, trues_cat)
        val_mase = Metrics.MASE(preds_cat, trues_cat)

        avg_train_rmse = np.mean(train_mw_losses)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1}/{args.epochs} | Time: {t_epoch_end - t_epoch_start:.2f}s | LR: {current_lr:.6f}")
        print(f"  Train RMSE: {avg_train_rmse:.4f} MW (Physical)")
        print(
            f"  >>> Valid Metrics: MAE: {val_mae:.4f} MW | RMSE: {val_rmse_metric:.4f} MW | R2: {val_r2:.4f} | MASE: {val_mase:.4f}")

        current_score = val_rmse_metric

        scheduler.step(current_score)

        if current_score < best_score:
            best_score = current_score
            torch.save(model.state_dict(), warmup_checkpoint)
            print(f"  >>> Warmup Best Saved (RMSE: {best_score:.4f} MW)")

        if current_lr < 1e-6:
            print("Learning rate too small, stopping warmup.")
            break

    print(f"\n[Phase 1 Complete] Warmup weights saved to: {warmup_checkpoint}")


if __name__ == '__main__':
    main()