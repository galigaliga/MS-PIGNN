import pandas as pd
import numpy as np
import os
import glob


class ConstraintChecker:
    def __init__(self, output_dir, graph_path):
        """
        :param output_dir: 包含 train.py 导出的3个csv的目录
        :param graph_path: 包含 gen.csv, ptdf.csv 的目录
        """
        self.output_dir = output_dir
        self.graph_path = graph_path

        # 基础浮点容差 (用于数值比较)
        self.tol = 1e-4

        # 1. 自动寻找文件
        self.pred_file = self._find_file("*_prediction.csv")
        self.true_file = self._find_file("ground_truth_power.csv")
        self.phys_file = self._find_file("ground_truth_physics_load.csv")

        # 2. 加载数据
        self.load_data()

        # 3. 加载静态参数
        self.load_static_data()

    def _find_file(self, pattern):
        files = glob.glob(os.path.join(self.output_dir, pattern))
        if not files:
            raise FileNotFoundError(f"Missing {pattern} in {self.output_dir}.")
        return max(files, key=os.path.getmtime)

    def load_data(self):
        print(f">>> Loading Data from: {self.output_dir}")
        print(f"    Pred (Model Output): {os.path.basename(self.pred_file)}")
        print(f"    True (Gurobi Gen):   {os.path.basename(self.true_file)}")
        print(f"    Phys (Aligned Load): {os.path.basename(self.phys_file)}")

        self.P_pred = pd.read_csv(self.pred_file).values
        self.P_true = pd.read_csv(self.true_file).values
        self.phys_data = pd.read_csv(self.phys_file).values

        # 长度对齐
        min_len = min(self.P_pred.shape[0], self.P_true.shape[0], self.phys_data.shape[0])
        self.P_pred = self.P_pred[:min_len]
        self.P_true = self.P_true[:min_len]
        self.phys_data = self.phys_data[:min_len]

        # 解析负荷 (前118列)
        self.ts_load = self.phys_data[:, 0:118]
        print(f"    Aligned Length: {min_len} time steps")

    def load_static_data(self):
        print(">>> Loading Static Data...")
        gen_df = pd.read_csv(os.path.join(self.graph_path, 'gen.csv'), header=0)

        self.gen_bus = gen_df.iloc[:, 1].values.astype(int) - 1
        self.P_min = gen_df.iloc[:, 2].values
        self.P_max = gen_df.iloc[:, 3].values
        self.R_up = gen_df.iloc[:, 4].values
        self.R_down = gen_df.iloc[:, 5].values

        self.num_gens = len(gen_df)

        # 加载启停时间参数 (MinUp, MinDown)
        # 注意：源数据单位是小时，需要转换为时间步 (x6)
        if gen_df.shape[1] > 7:
            self.MinUp = gen_df.iloc[:, 6].values * 6
            self.MinDown = gen_df.iloc[:, 7].values * 6
            print(f"    MinUp/MinDown loaded and converted (x6). Max MinUp: {np.max(self.MinUp)} steps.")
        else:
            print("    [WARNING] MinUp/MinDown columns not found, defaulting to 6 steps.")
            self.MinUp = np.ones(self.num_gens) * 6
            self.MinDown = np.ones(self.num_gens) * 6

        # 支路参数
        branch_df = pd.read_csv(os.path.join(self.graph_path, 'branch.csv'), header=0)
        self.PL_max = branch_df.iloc[:, 7].values

        # PTDF
        ptdf_path = os.path.join(self.graph_path, 'ptdf.csv')
        if os.path.exists(ptdf_path):
            self.ptdf = pd.read_csv(ptdf_path, header=None).values
        else:
            self.ptdf = None
            print("    Warning: PTDF not found, skipping flow check.")

        self.num_buses = 118

    # =========================================================================
    # 1. 机组出力上下限检查
    # =========================================================================
    def check_gen_limits(self):
        print("\n[1] Checking Generation Limits (Detailed)...")
        P = self.P_pred
        total_points = P.size

        # A. 越上限
        viol_max_mask = P > (self.P_max + self.tol)
        count_max = np.sum(viol_max_mask)

        # B. 越下限 (开机状态下 < Min)
        # 此时 P_pred 已经很干净，> tol 即为开机
        is_on = P > self.tol
        below_min = P < (self.P_min - self.tol)
        viol_min_mask = is_on & below_min
        count_min = np.sum(viol_min_mask)

        # C. 统计
        total_viol = count_max + count_min
        rate = total_viol / total_points

        print(f"    Total Violations      : {total_viol} ({rate:.4%})")
        return rate

    # =========================================================================
    # 2. 爬坡率检查
    # =========================================================================
    def check_ramp_rates(self):
        print("\n[2] Checking Ramp Rates (Steady State)...")

        diff_P = np.diff(self.P_pred, axis=0)
        is_on = self.P_pred > self.tol

        # 只检查稳态爬坡 (t 和 t+1 都在线)
        is_continuous_run = is_on[:-1, :] & is_on[1:, :]

        viol_up = (diff_P > (self.R_up + self.tol)) & is_continuous_run
        viol_down = (diff_P < (-self.R_down - self.tol)) & is_continuous_run

        count = np.sum(viol_up) + np.sum(viol_down)
        total_continuous_points = np.sum(is_continuous_run)

        rate = count / total_continuous_points if total_continuous_points > 0 else 0.0

        print(f"    Violations: {count} / {total_continuous_points} continuous steps ({rate:.4%})")
        return rate

    # =========================================================================
    # 3. 功率平衡检查 (详细版)
    # =========================================================================
    def check_power_balance(self, points_per_day=144):
        print("\n[3] Checking Power Balance (Thermal Only)...")

        sum_gen = np.sum(self.P_pred, axis=1)
        sum_load = np.sum(self.ts_load, axis=1)

        mismatch_mw = sum_gen - sum_load
        abs_mismatch_mw = np.abs(mismatch_mw)
        relative_error = abs_mismatch_mw / (sum_load + 1e-6)

        eng_threshold = 0.05  # 5% 阈值

        # --- Part A: 按天统计 ---
        print(f"    >>> A. Daily Report (Days with violations)")
        print(
            f"    {'Day':<5} | {'Max Error (MW)':<15} | {'Mean Error (MW)':<16} | {'Viol Rate (>5%)':<16} | {'Status'}")
        print(f"    {'-' * 5}-|-{'-' * 15}-|-{'-' * 16}-|-{'-' * 16}-|-{'-' * 10}")

        total_points = len(mismatch_mw)
        num_days = int(np.ceil(total_points / points_per_day))
        days_with_issues = 0

        for d in range(num_days):
            start_idx = d * points_per_day
            end_idx = min((d + 1) * points_per_day, total_points)

            day_abs_err = abs_mismatch_mw[start_idx:end_idx]
            day_rel_err = relative_error[start_idx:end_idx]
            if len(day_abs_err) == 0: continue

            d_max_mw = np.max(day_abs_err)
            d_mean_mw = np.mean(day_abs_err)
            d_viol_count = np.sum(day_rel_err > eng_threshold)

            if d_viol_count > 0:
                days_with_issues += 1
                status = "VIOLATION" if d_viol_count > 0 else "High Error"
                d_viol_rate = d_viol_count / len(day_rel_err)
                print(f"    {d + 1:<5} | {d_max_mw:>15.4f} | {d_mean_mw:>16.4f} | {d_viol_rate:>15.2%}  | {status}")

        if days_with_issues == 0:
            print(f"    [Excellent] No days triggered significant violations.")

        # --- Part B: 按时间段统计 ---
        print(f"\n    >>> B. Worst Time Slots (Aggregated across all days)")
        print(f"    {'Time':<10} | {'Viol Rate':<12} | {'Mean Error (MW)':<18} | {'Max Error (MW)'}")
        print(f"    {'-' * 10}-|-{'-' * 12}-|-{'-' * 18}-|-{'-' * 15}")

        num_full_days = total_points // points_per_day
        eff_len = num_full_days * points_per_day

        if num_full_days > 0:
            reshaped_abs_err = abs_mismatch_mw[:eff_len].reshape(num_full_days, points_per_day)
            reshaped_rel_err = relative_error[:eff_len].reshape(num_full_days, points_per_day)

            slot_viol_rate = np.mean(reshaped_rel_err > eng_threshold, axis=0)
            slot_mean_err = np.mean(reshaped_abs_err, axis=0)
            slot_max_err = np.max(reshaped_abs_err, axis=0)

            slot_stats = []
            for t in range(points_per_day):
                h = (t * 10) // 60
                m = (t * 10) % 60
                time_str = f"{h:02d}:{m:02d}"
                slot_stats.append({
                    'time': time_str,
                    'viol_rate': slot_viol_rate[t],
                    'mean_err': slot_mean_err[t],
                    'max_err': slot_max_err[t]
                })

            sorted_slots = sorted(slot_stats, key=lambda x: (x['viol_rate'], x['max_err']), reverse=True)

            count_printed = 0
            for s in sorted_slots:
                if s['viol_rate'] > 0 or s['max_err'] > 30.0:
                    print(
                        f"    {s['time']:<10} | {s['viol_rate']:>11.2%} | {s['mean_err']:>18.4f} | {s['max_err']:>14.4f}")
                    count_printed += 1
                if count_printed >= 10: break

            if count_printed == 0:
                print("    [Excellent] Time slot errors are within acceptable range.")

        return np.mean(relative_error > eng_threshold)

    # =========================================================================
    # 4. 线路潮流检查
    # =========================================================================
    def check_branch_flow(self):
        print("\n[4] Checking Branch Flow Limits...")
        if self.ptdf is None: return 0.0

        P_inj = np.zeros((self.P_pred.shape[0], self.num_buses))
        P_inj -= self.ts_load
        for g_idx in range(self.num_gens):
            bus = self.gen_bus[g_idx]
            if 0 <= bus < self.num_buses:
                P_inj[:, bus] += self.P_pred[:, g_idx]

        Flow = self.ptdf @ P_inj.T
        Flow = Flow.T
        Limit = self.PL_max.reshape(1, -1)
        viol = np.abs(Flow) > (Limit + self.tol)

        count = np.sum(viol)
        rate = count / Flow.size
        print(f"    Violations: {count} ({rate:.4%})")
        return rate

    # =========================================================================
    # 5. 启停约束检查 (最终验证)
    # =========================================================================
    def check_uc_constraints(self):
        print("\n[5] Checking Min Up/Down Time Constraints (Verification)...")

        is_on = (self.P_pred > self.tol).astype(int)

        T, num_gens = is_on.shape
        total_up_viol = 0
        total_down_viol = 0
        total_actions = 0

        diff_status = np.diff(is_on, axis=0)

        for g in range(num_gens):
            min_up = int(self.MinUp[g])
            min_down = int(self.MinDown[g])

            # 启动时刻
            start_indices = np.where(diff_status[:, g] == 1)[0] + 1
            # 停机时刻
            stop_indices = np.where(diff_status[:, g] == -1)[0] + 1

            total_actions += (len(start_indices) + len(stop_indices))

            # 检查 Min Up
            for t_start in start_indices:
                check_end = min(t_start + min_up, T)
                if not np.all(is_on[t_start:check_end, g] == 1):
                    total_up_viol += 1

            # 检查 Min Down
            for t_stop in stop_indices:
                check_end = min(t_stop + min_down, T)
                if not np.all(is_on[t_stop:check_end, g] == 0):
                    total_down_viol += 1

        rate = 0.0
        if total_actions > 0:
            rate = (total_up_viol + total_down_viol) / total_actions

        print(f"    Min Up Violations   : {total_up_viol}")
        print(f"    Min Down Violations : {total_down_viol}")
        print(f"    Total UC Violations : {total_up_viol + total_down_viol} / {total_actions} actions ({rate:.4%})")

        return rate

    def run_full_pipeline(self):
        print("=========================================")
        print(f"Running Full Validation Pipeline (Greedy Repair)")
        print("=========================================")

        r1 = self.check_gen_limits()
        r2 = self.check_ramp_rates()
        r3 = self.check_power_balance(points_per_day=144)
        r4 = self.check_branch_flow()
        r5 = self.check_uc_constraints()

        print("\n-----------------------------------------")
        print("Summary (Post-Repair):")
        print(f"Gen Limits : {r1:.2%}")
        print(f"Ramp Rates : {r2:.2%}")
        print(f"Balance    : {r3:.2%}")
        print(f"Branch Flow: {r4:.2%}")
        print(f"UC (Up/Dn) : {r5:.2%}")


if __name__ == "__main__":
    # 配置路径
    OUTPUT_DIR = "./output/MSPIGNN/"
    GRAPH_DIR = "./ieee118new"

    checker = ConstraintChecker(OUTPUT_DIR, GRAPH_DIR)
    checker.run_full_pipeline()