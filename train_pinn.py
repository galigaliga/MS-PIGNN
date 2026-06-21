import torch
import torch.nn as nn
import torch.optim as optim
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


class ALM_Optimizer:
    def __init__(self, device, mu_init, beta, max_mu, num_timesteps, tolerance=0.01):
        self.device = device
        # 维护两个 mu: 一个用于平衡(bal)，一个用于潮流(flow)
        self.mu_bal = mu_init
        self.mu_flow = mu_init

        self.beta = beta
        self.max_mu = max_mu
        self.tolerance = tolerance

        self.lambda_mult = torch.zeros(num_timesteps, device=device, requires_grad=False)

    def update_params(self, avg_residual, current_epoch_bal_mse, current_epoch_flow_viol):
        with torch.no_grad():
            self.lambda_mult += self.mu_bal * avg_residual
            current_bal_rmse = np.sqrt(current_epoch_bal_mse)
            if current_bal_rmse > self.tolerance:
                self.mu_bal = min(self.mu_bal * self.beta, self.max_mu)
            if current_epoch_flow_viol > self.tolerance:
                self.mu_flow = min(self.mu_flow * self.beta, self.max_mu)
        print(
            f"  [ALM Update] mu_bal: {self.mu_bal:.4f} | mu_flow: {self.mu_flow:.4f} | Mean Lambda: {self.lambda_mult.mean().item():.4f}")


def differentiable_inverse_transform_y(tensor, mean, scale, device):
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=device, dtype=torch.float32)
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=device, dtype=torch.float32)
    return tensor * scale + mean

def state_fix(flat_pred, p_min_arr, r_down_arr, min_up_arr, min_down_arr):
    T, N = flat_pred.shape
    thresholds = p_min_arr * 0.5
    raw_on = (flat_pred > thresholds).astype(int)

    fixed_status = np.zeros((T, N), dtype=int)
    repaired_P = flat_pred.copy()
    changes_count = 0

    for g in range(N):
        min_up = int(min_up_arr[g])
        min_down = int(min_down_arr[g])
        p_min = p_min_arr[g]
        r_down = r_down_arr[g]

        current_state = raw_on[0, g]
        locked_until = 0
        fixed_status[0, g] = current_state

        for t in range(1, T):
            if t < locked_until:
                fixed_status[t, g] = current_state
            else:
                model_intent = raw_on[t, g]
                if model_intent != current_state:
                    if model_intent == 1:
                        current_state = 1
                        locked_until = t + min_up
                    else:
                        current_state = 0
                        locked_until = t + min_down
                fixed_status[t, g] = current_state

        if fixed_status[0, g] == 1 and raw_on[0, g] == 0:
            repaired_P[0, g] = max(repaired_P[0, g], p_min)
            changes_count += 1
        elif fixed_status[0, g] == 0 and raw_on[0, g] == 1:
            repaired_P[0, g] = 0.0
            changes_count += 1

        for t in range(1, T):
            if fixed_status[t, g] == 0:
                if repaired_P[t, g] > 0:
                    repaired_P[t, g] = 0.0
                    changes_count += 1
            else:
                if raw_on[t, g] == 1:
                    pass
                else:
                    prev_p = repaired_P[t - 1, g]
                    max_drop = prev_p - r_down
                    repaired_P[t, g] = max(p_min, max_drop)
                    changes_count += 1

    return repaired_P


def main():
    args = get_config()
    setup_seed(args.seed)
    device = torch.device(args.device)

    # 路径配置
    output_base_dir = './output'
    model_save_dir = os.path.join(output_base_dir, args.model_name)
    warmup_checkpoint = os.path.join(model_save_dir, 'warmup_checkpoint.pth')

    if not os.path.exists(warmup_checkpoint):
        raise FileNotFoundError(f"Missing warmup checkpoint: {warmup_checkpoint}. Please run Phase 1 first.")

    print(f"\n==================================================")
    print(f"   PHASE 2: ALM OPTIMIZATION (Physical MW Loss)")
    print(f"==================================================")
    print(f"Loading Weights from: {warmup_checkpoint}")

    # 1. 加载数据
    train_set, train_loader = get_loader(args, flag='train')
    valid_set, valid_loader = get_loader(args, flag='val')
    test_set, test_loader = get_loader(args, flag='test')

    args.input_size = train_set.data_x.shape[1]
    args.num_nodes = 118
    args.output_size = train_set.power_labels.shape[1]

    print(f"[System Detected] Nodes: {args.num_nodes}, Generators: {args.output_size}")

    # 2. 图与特征映射
    adj_matrix, _ = load_graph_data(args.graph_path, num_nodes=args.num_nodes, device=device)
    args.adj = adj_matrix
    input_mask = build_input_mapping(data_path=args.data_path, graph_path=args.graph_path, num_nodes=args.num_nodes,
                                     device=device)
    args.input_mask = input_mask

    y_mean = train_set.y_mean
    y_scale = train_set.y_scale
    if hasattr(test_set, 'export_ground_truth'): test_set.export_ground_truth(model_save_dir)
    args.y_mean = torch.tensor(y_mean, device=device).float()
    args.y_scale = torch.tensor(y_scale, device=device).float()

    # =======================================================
    # 【加载物理参数】(直接物理值，不归一化)
    # =======================================================
    print(f">>> Loading Physical Parameters (Raw MW)...")

    gen_path = os.path.join(args.graph_path, 'gen.csv')
    gen_df = pd.read_csv(gen_path)

    if len(gen_df) != args.output_size:
        gen_df = gen_df.iloc[:args.output_size]

    p_min_raw = gen_df.iloc[:, 2].values.astype(np.float32)
    p_max_raw = gen_df.iloc[:, 3].values.astype(np.float32)
    ramp_up_raw = gen_df.iloc[:, 4].values.astype(np.float32)
    ramp_down_raw = gen_df.iloc[:, 5].values.astype(np.float32)

    args.p_min = torch.tensor(p_min_raw, device=device).float()
    args.p_max = torch.tensor(p_max_raw, device=device).float()
    args.ramp_up = torch.tensor(ramp_up_raw, device=device).float()
    args.ramp_down = torch.tensor(ramp_down_raw, device=device).float()

    # Gen to Node Mapping
    if 'bus_id' in gen_df.columns:
        gen_node_idx = gen_df['bus_id'].values - 1
    else:
        gen_node_idx = gen_df.iloc[:, 1].values - 1
    args.gen_map_index = torch.tensor(gen_node_idx, device=device).long()

    # PTDF & Limits
    ptdf_path = os.path.join(args.graph_path, 'ptdf.csv')
    branch_path = os.path.join(args.graph_path, 'branch.csv')

    if os.path.exists(ptdf_path) and os.path.exists(branch_path):
        print("  Loading PTDF...")
        try:
            ptdf_df = pd.read_csv(ptdf_path, header=None)
        except:
            ptdf_df = pd.read_csv(ptdf_path, header=0)
        args.full_ptdf = torch.tensor(ptdf_df.values, device=device).float()

        try:
            branch_df = pd.read_csv(branch_path, header=0)
        except:
            branch_df = pd.read_csv(branch_path, header=None)
        args.full_limits = torch.tensor(branch_df.iloc[:, 7].values.astype(np.float32), device=device).float()
        print(f"  Loaded PTDF: {args.full_ptdf.shape}")
    else:
        print("  [Warning] ptdf.csv not found.")

    # 4. 初始化模型
    model = MSPIGNN.Model(args).to(device)
    model.load_state_dict(torch.load(warmup_checkpoint, weights_only=True), strict=False)
    print(">>> Phase 1 Weights Loaded. Starting ALM Training...")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.5)
    alm_opt = ALM_Optimizer(device, args.mu_init, args.beta, max_mu=200.0, num_timesteps=args.pred_len, tolerance=0.01)

    save_file = os.path.join(model_save_dir, 'alm_best_model.pth')
    best_score = float('inf')

    print("\n--- Start ALM Training ---")

    for epoch in range(args.epochs):
        t_epoch_start = time.time()
        model.train()

        train_task_losses = []
        train_phy_losses_mse = []
        train_flow_violations = []
        train_bias_mw = []
        epoch_residuals_scaled = []

        for batch_idx, (batch_x, batch_future, batch_y, batch_phys) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_future = batch_future.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_phys = batch_phys.float().to(device)

            optimizer.zero_grad()

            pred_mw_final, gate_out = model(batch_x, batch_future)

            batch_y_mw = differentiable_inverse_transform_y(batch_y, args.y_mean, args.y_scale, device)

            loss_task_mse_mw = criterion(pred_mw_final, batch_y_mw)

            loss_task = torch.sqrt(loss_task_mse_mw + 1e-6) / 10.0

            # Balance Loss
            load_mw = batch_phys[:, :, :args.num_nodes]
            B, T, _ = pred_mw_final.shape
            gen_mw_nodal = torch.zeros(B, T, args.num_nodes, device=device)
            idx_expanded = args.gen_map_index.view(1, 1, -1).expand(B, T, -1)
            gen_mw_nodal.scatter_(2, idx_expanded, pred_mw_final)

            net_injection_mw = gen_mw_nodal - load_mw
            sys_imbalance = torch.sum(net_injection_mw, dim=2)
            sum_load = torch.sum(load_mw, dim=2)
            residual_norm = sys_imbalance / (sum_load + 1e-6)
            loss_bal_mse = torch.mean(residual_norm ** 2)
            loss_bal_rmse = torch.sqrt(loss_bal_mse + 1e-6)

            # Flow Loss (Full Check)
            if hasattr(model, 'trans_layer'):
                mean_flow_viol, _ = model.trans_layer(net_injection_mw)
                loss_flow = mean_flow_viol
            else:
                loss_flow = torch.tensor(0.0, device=device)

            loss_lagrangian = torch.mean(alm_opt.lambda_mult * residual_norm)
            loss_total = loss_task + (1.0 * alm_opt.mu_bal * loss_bal_rmse) + (
                    1.0 * alm_opt.mu_flow * loss_flow) + loss_lagrangian

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_task_losses.append(loss_task.item())
            train_phy_losses_mse.append(loss_bal_mse.item())
            train_flow_violations.append(loss_flow.item())
            train_bias_mw.append(torch.mean(sys_imbalance).item())
            epoch_residuals_scaled.append(torch.mean(residual_norm, dim=0).detach())

        # ALM Update
        avg_epoch_res_scaled = torch.stack(epoch_residuals_scaled).mean(dim=0)
        current_epoch_bal_mse = np.mean(train_phy_losses_mse)
        current_epoch_flow_viol = np.mean(train_flow_violations)
        alm_opt.update_params(avg_epoch_res_scaled, current_epoch_bal_mse, current_epoch_flow_viol)

        # Validation
        model.eval()
        val_preds_mw = []
        val_trues_mw = []
        with torch.no_grad():
            for batch_x, batch_future, batch_y, _ in valid_loader:
                batch_x = batch_x.float().to(device);
                batch_future = batch_future.float().to(device);
                batch_y = batch_y.float().to(device)

                pred_mw_final, _ = model(batch_x, batch_future)

                true_mw = differentiable_inverse_transform_y(batch_y, args.y_mean, args.y_scale, device)
                val_preds_mw.append(pred_mw_final.cpu().numpy());
                val_trues_mw.append(true_mw.cpu().numpy())

        t_epoch_end = time.time()
        preds_cat = np.concatenate(val_preds_mw, axis=0)
        trues_cat = np.concatenate(val_trues_mw, axis=0)

        val_mae = Metrics.MAE(preds_cat, trues_cat)
        val_rmse_metric = Metrics.RMSE(preds_cat, trues_cat)
        val_r2 = Metrics.R2(preds_cat, trues_cat)
        val_mase = Metrics.MASE(preds_cat, trues_cat)

        avg_train_rmse = np.mean(train_task_losses)
        avg_phy_norm_rmse = np.sqrt(current_epoch_bal_mse)
        avg_flow_viol = np.mean(train_flow_violations)
        avg_bias = np.mean(train_bias_mw)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1}/{args.epochs} | Time: {t_epoch_end - t_epoch_start:.2f}s | LR: {current_lr:.6f}")
        print(
            f"  Train Task RMSE: {avg_train_rmse * 10:.5f} | Phy(Bal) RMSE: {avg_phy_norm_rmse:.4f} | Flow Viol: {avg_flow_viol:.4f} | Bias: {avg_bias:.2f} MW")
        print(
            f"  >>> Real Metrics: MAE: {val_mae:.4f} MW | RMSE: {val_rmse_metric:.4f} | R2: {val_r2:.4f} | MASE: {val_mase:.4f}")

        total_phy_error = avg_phy_norm_rmse + 0.1 * avg_flow_viol
        current_score = val_rmse_metric + 10.0 * total_phy_error
        if current_score < best_score:
            best_score = current_score
            torch.save(model.state_dict(), save_file)
            print(
                f"  >>> ALM Best Model Saved (Score: {current_score:.4f} | RMSE: {val_rmse_metric:.4f} | Phy_Total: {total_phy_error:.4f})")

        if current_lr < 1e-6:
            print(f"\n[Terminated] Learning rate {current_lr:.8f} is below threshold.")
            break

    # Final Test
    print("\n--- Final Test & Saving Aligned Data ---")
    model.load_state_dict(torch.load(save_file, weights_only=True))
    model.eval()

    preds = [];
    trues = [];
    phys_list = []
    if 'cuda' in args.device: torch.cuda.synchronize()
    t_start = time.time()

    with torch.no_grad():
        for batch_x, batch_future, batch_y, batch_phys in test_loader:
            batch_x = batch_x.float().to(device);
            batch_future = batch_future.float().to(device);
            batch_y = batch_y.float().to(device)

            pred_mw_final, _ = model(batch_x, batch_future)

            true_mw = differentiable_inverse_transform_y(batch_y, args.y_mean, args.y_scale, device)
            preds.append(pred_mw_final.cpu().numpy());
            trues.append(true_mw.cpu().numpy());
            phys_list.append(batch_phys.numpy())

    if 'cuda' in args.device: torch.cuda.synchronize()
    t_end = time.time()
    preds = np.concatenate(preds, axis=0);
    trues = np.concatenate(trues, axis=0);
    phys_arr = np.concatenate(phys_list, axis=0)

    total_samples = preds.shape[0];
    total_time = t_end - t_start;
    avg_time_per_sample = (total_time / total_samples) * 1000
    mae = Metrics.MAE(preds, trues);
    rmse = Metrics.RMSE(preds, trues);
    r2 = Metrics.R2(preds, trues);
    mase = Metrics.MASE(preds, trues)

    print(f"\n[Accuracy Metrics]\nMAE : {mae:.4f} MW\nRMSE: {rmse:.4f} MW\nMASE: {mase:.4f}\nR2  : {r2:.4f}")
    print(
        f"\n[Speed Metrics]\nTotal Inference Time   : {total_time:.4f} s\nTime per Dispatch Cycle: {avg_time_per_sample:.4f} ms")

    print(f"\n[Stitching Rolling Data]...")
    test_step = test_set.step
    N, T, D = preds.shape
    flat_pred = preds[:, :test_step, :].reshape(-1, D)
    flat_true = trues[:, :test_step, :].reshape(-1, D)
    flat_phys_load = phys_arr[:, :test_step, :args.num_nodes].reshape(-1, args.num_nodes)

    min_up_raw = gen_df.iloc[:, 6].values.astype(np.float32) * 6
    min_down_raw = gen_df.iloc[:, 7].values.astype(np.float32) * 6

    flat_pred_repaired = state_fix(
        flat_pred=flat_pred,
        p_min_arr=p_min_raw,
        r_down_arr=ramp_down_raw,
        min_up_arr=min_up_raw,
        min_down_arr=min_down_raw
    )

    col_names_gen = [f'Unit_{i + 1}' for i in range(D)]
    col_names_load = [f'Load_Bus{i + 1}' for i in range(args.num_nodes)]

    pred_csv_path = os.path.join(model_save_dir, f'{args.model_name}_prediction.csv')
    pd.DataFrame(flat_pred_repaired, columns=col_names_gen).to_csv(pred_csv_path, index=False, float_format='%.4f')
    pred_raw_csv_path = os.path.join(model_save_dir, f'{args.model_name}_prediction_raw.csv')
    pd.DataFrame(flat_pred, columns=col_names_gen).to_csv(pred_raw_csv_path, index=False, float_format='%.4f')
    true_csv_path = os.path.join(model_save_dir, 'ground_truth_power.csv')
    pd.DataFrame(flat_true, columns=col_names_gen).to_csv(true_csv_path, index=False, float_format='%.4f')
    phys_csv_path = os.path.join(model_save_dir, 'ground_truth_physics_load.csv')
    pd.DataFrame(flat_phys_load, columns=col_names_load).to_csv(phys_csv_path, index=False, float_format='%.4f')

    print(f"-> Saved Prediction to: {pred_csv_path}")
    print(f"-> Saved Aligned GT to: {true_csv_path}")
    print(f"-> Saved Aligned Physics (Load) to: {phys_csv_path}")
    print("Done.")


if __name__ == '__main__':
    main()