import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# --- 1. 块状加载函数 (Unit-Major) ---
def load_and_reshape_matrix(filepath, num_nodes, num_days=365, num_timesteps=144, start_col=0):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, header=0)
    data = df.values[:, start_col:]
    data = data[:, :num_days]

    # [Node, Time, Day] -> [Day, Time, Node] -> [Total_Time, Node]
    data = data.reshape(num_nodes, num_timesteps, num_days)
    data = data.transpose(2, 1, 0)
    data = data.reshape(-1, num_nodes)
    return data


# --- 2. 交织加载函数 (Time-Major) ---
def load_interleaved_label(filepath, num_units, num_days=365, num_timesteps=144, stride=None, start_col=0):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if stride is None:
        stride = num_units

    df = pd.read_csv(filepath, header=0)
    raw_data = df.values[:, start_col:]
    raw_data = raw_data[:, :num_days]

    required_rows = num_timesteps * stride
    if raw_data.shape[0] < required_rows:
        pass

    raw_data = raw_data[:required_rows, :]

    # [Time, Stride, Day] -> [Day, Time, Unit] -> [Total_Time, Unit]
    data = raw_data.reshape(num_timesteps, stride, num_days)
    data = data[:, :num_units, :]
    data = data.transpose(2, 0, 1)
    data = data.reshape(-1, num_units)
    return data


class StandardDataset(Dataset):
    def __init__(self, args, flag='train'):
        self.seq_len = args.seq_len  # 历史长度 (144)
        self.pred_len = args.pred_len  # 预测长度 (24)
        self.flag = flag

        if self.flag == 'test':
            self.step = 6
        else:
            self.step = 1

        NUM_DAYS = 365
        NUM_TIMESTEPS = 144
        path = args.data_path

        if self.flag == 'train':
            print(f"[{flag.upper()}] Loading data (Thermal Only Mode with Future Load)...")

        # 1. Load (原始负荷数据)
        self.load_feat = load_and_reshape_matrix(
            os.path.join(path, 'all_busloads_144.csv'),
            num_nodes=118, num_days=NUM_DAYS, num_timesteps=NUM_TIMESTEPS,
            start_col=0
        )

        # 2. Power (原始发电标签)
        self.power_labels = load_interleaved_label(
            os.path.join(path, 'power.csv'),
            num_units=54, num_days=NUM_DAYS, num_timesteps=NUM_TIMESTEPS,
            stride=54, start_col=0
        )

        # 【新增 1】计算系统总负荷 (用于作为未来特征)
        # [Total_Time, 118] -> [Total_Time, 1]
        self.total_load = np.sum(self.load_feat, axis=1, keepdims=True)

        # 物理 Scale (用于日志打印)
        self.phys_scale = np.std(self.total_load)
        if self.phys_scale < 1e-3: self.phys_scale = 1.0

        if self.flag == 'train':
            print(f"  [Physics Norm] Scale Factor: {self.phys_scale:.4f}")

        # 5. 构建原始数据
        # Input X: 历史 Load + Power
        raw_data_x = np.concatenate([self.load_feat, self.power_labels], axis=1)
        # Output Y: Power
        raw_data_y = self.power_labels
        # 【新增 2】Future Input: 未来总负荷
        raw_data_future = self.total_load

        # 6. Physics Data (用于 Loss 和 校验)
        self.raw_phys_data = np.concatenate([
            self.load_feat,
            self.power_labels
        ], axis=1)

        # 归一化器
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.scaler_future = StandardScaler()  # 【新增 3】

        total_len = len(raw_data_x)
        train_len = int(total_len * 0.8)

        # Fit (仅在训练集)
        self.scaler_x.fit(raw_data_x[:train_len])
        self.scaler_y.fit(raw_data_y[:train_len])
        self.scaler_future.fit(raw_data_future[:train_len])  # Fit

        # Transform (全部数据)
        self.data_x = self.scaler_x.transform(raw_data_x)
        self.data_y = self.scaler_y.transform(raw_data_y)
        self.data_future = self.scaler_future.transform(raw_data_future)  # Transform

        self.y_mean = self.scaler_y.mean_
        self.y_scale = self.scaler_y.scale_

        # 切分逻辑
        if self.flag == 'test':
            self.test_days = 36
            self.target_points = self.test_days * 144
            self.num_samples = self.target_points // self.step
            last_sample_idx = (self.num_samples - 1) * self.step
            required_len = last_sample_idx + self.seq_len + self.pred_len

            start_idx = self.data_x.shape[0] - required_len

            self.data_x = self.data_x[start_idx:]
            self.data_y = self.data_y[start_idx:]
            self.data_future = self.data_future[start_idx:]  # 【新增同步切分】
            self.raw_phys_data = self.raw_phys_data[start_idx:]
            print(f"  [Test Mode] Robust Setup Applied. Sliced from: {start_idx}")

        elif self.flag == 'train':
            self.data_x = self.data_x[:train_len]
            self.data_y = self.data_y[:train_len]
            self.data_future = self.data_future[:train_len]  # 【新增同步切分】
            self.raw_phys_data = self.raw_phys_data[:train_len]
            self.num_samples = (len(self.data_x) - self.seq_len - self.pred_len + 1) // self.step

        elif self.flag == 'val':
            val_len = int(total_len * 0.1)
            start_val = train_len - self.seq_len
            end_val = train_len + val_len

            self.data_x = self.data_x[start_val: end_val]
            self.data_y = self.data_y[start_val: end_val]
            self.data_future = self.data_future[start_val: end_val]  # 【新增同步切分】
            self.raw_phys_data = self.raw_phys_data[start_val: end_val]
            self.num_samples = (len(self.data_x) - self.seq_len - self.pred_len + 1) // self.step

    def __getitem__(self, index):
        idx = index * self.step
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        # 1. 历史状态 X [Seq_Len, Features]
        seq_x = self.data_x[s_begin: s_end]

        # 2. 【核心新增】预测视窗内的未来负荷 [Pred_Len, 1]
        # 注意：取值范围是 r_begin 到 r_end (即未来时刻)
        seq_future_load = self.data_future[r_begin: r_end]

        # 3. 标签 Y [Pred_Len, Units]
        seq_y = self.data_y[r_begin: r_end]

        # 4. 原始物理数据 [Pred_Len, Phys_Feats]
        seq_phys = self.raw_phys_data[r_begin: r_end]

        # 返回 4 个 Tensor
        return (torch.FloatTensor(seq_x),
                torch.FloatTensor(seq_future_load),
                torch.FloatTensor(seq_y),
                torch.FloatTensor(seq_phys))

    def __len__(self):
        return self.num_samples

    def inverse_transform(self, data):
        return self.scaler_y.inverse_transform(data)

    def export_ground_truth(self, save_dir):
        import pandas as pd
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        print(f"[{self.flag.upper()}] Exporting Aligned Ground Truth to {save_dir}...")

        col_names = []
        col_names.extend([f'Load_{i + 1}' for i in range(118)])
        col_names.extend([f'Unit_{i + 1}' for i in range(54)])

        save_path = os.path.join(save_dir, 'ground_truth_aligned.csv')
        pd.DataFrame(self.raw_phys_data, columns=col_names).to_csv(save_path, index=False, float_format='%.6f')
        print(f"-> Exported {save_path}")


def get_loader(args, flag):
    dataset = StandardDataset(args, flag)
    shuffle = True if flag == 'train' else False
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)
    return dataset, loader