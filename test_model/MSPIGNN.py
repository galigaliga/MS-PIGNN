import torch
import torch.nn as nn
import numpy as np
import argparse

# 引入约束层
from .constraints import CapacityConstraintLayer, RampClamp, FullNetworkTransmissionLayer


# =========================================================================
# 1. 新一代时序分解组件 (基于 SegRNN 分块思想 + 多尺度)
# =========================================================================

class SegmentedTrendLayer(nn.Module):
    """
    基于 SegRNN 思想的分块递归层 (Segmented Recurrent Layer)
    """

    def __init__(self, segment_len, input_dim=1):
        super(SegmentedTrendLayer, self).__init__()
        self.segment_len = segment_len
        # 使用 GRU 模拟 EMA 的递归特性
        self.rnn = nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=True)
        # 用于调整幅度的线性层
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        B, N, T = x.shape
        S = self.segment_len

        # 1. Padding: 确保 T 能被 S 整除
        if T % S != 0:
            pad_size = S - (T % S)
            x = torch.nn.functional.pad(x, (0, pad_size), mode='replicate')
        else:
            pad_size = 0

        T_pad = x.shape[-1]
        Num_Segs = T_pad // S

        # 2. Reshape & Segmentation
        x_seg = x.reshape(B * N, Num_Segs, S)

        # 3. 块内聚合
        seg_means = x_seg.mean(dim=-1, keepdim=True)

        # 4. 块间递归
        trend_coarse, _ = self.rnn(seg_means)

        # 5. 上采样 & 调整
        trend_coarse = self.linear(trend_coarse)
        trend_fine = trend_coarse.repeat(1, 1, S).reshape(B, N, T_pad)

        # 6. 去除 Padding
        if pad_size > 0:
            trend_fine = trend_fine[..., :-pad_size]

        return trend_fine


class MultiScaleDecomp(nn.Module):
    """
    多尺度重构分解层 (Multi-Scale Decomposition)
    """

    def __init__(self, input_dim, scales=[6, 24, 48]):
        super(MultiScaleDecomp, self).__init__()
        self.scales = scales
        self.trend_layers = nn.ModuleList([
            SegmentedTrendLayer(segment_len=s, input_dim=input_dim)
            for s in scales
        ])
        self.fusion = nn.Linear(len(scales), 1)

    def forward(self, x):
        trends = []
        for layer in self.trend_layers:
            t = layer(x)
            trends.append(t.unsqueeze(-1))

        # [B, N, T, Num_Scales]
        trends_stack = torch.cat(trends, dim=-1)
        # [B, N, T]
        trend_final = self.fusion(trends_stack).squeeze(-1)

        seasonal = x - trend_final
        return seasonal, trend_final


# =========================================================================
# 2. 物理嵌入与图组件 (稀疏优化版)
# =========================================================================

class SparsePhysicsEmbedding(nn.Module):
    def __init__(self, in_features, out_features, mask):
        super(SparsePhysicsEmbedding, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        coo = mask.to_sparse()
        indices = coo.indices()
        self.register_buffer('indices', indices)

        self.nnz = indices.shape[1]
        self.values = nn.Parameter(torch.Tensor(self.nnz))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = np.sqrt(2.0 / 5.0)
        nn.init.normal_(self.values, 0, std)
        bound = 1 / np.sqrt(self.in_features) if self.in_features > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        B, T, In = x.shape
        Out = self.out_features
        x_flat = x.reshape(B * T, In).t()
        w_sparse = torch.sparse_coo_tensor(
            self.indices, self.values, size=(Out, In), device=x.device
        )
        out_flat = torch.sparse.mm(w_sparse, x_flat)
        out_flat = out_flat + self.bias.unsqueeze(1)
        out = out_flat.reshape(Out, B, T).permute(1, 2, 0)
        return out


class FastGCNLayer(nn.Module):
    def __init__(self, adj):
        super(FastGCNLayer, self).__init__()
        self.adj = adj
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, N, T = x.shape
        x_reshaped = x.permute(1, 0, 2).reshape(N, -1)
        out = torch.sparse.mm(self.adj, x_reshaped)
        out = out.reshape(N, B, T).permute(1, 0, 2)
        return out * self.weight + x


# =========================================================================
# 3. 主模型 (物理空间约束适配版 + 新分解组件)
# =========================================================================

class Model(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Model, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

        self.num_nodes = getattr(args, 'num_nodes', 118)
        self.output_size = getattr(args, 'output_size', 54)
        self.hidden_dim = getattr(args, 'node_hidden', 256)

        print(f"[GNN Init] System Scale: Nodes={self.num_nodes}, Gens={self.output_size}")

        if hasattr(args, 'adj'):
            self.adj = args.adj
        else:
            raise ValueError("Adj matrix missing!")

        input_mask = getattr(args, 'input_mask', None)
        if input_mask is None: raise ValueError("Input Mask missing!")

        # --- 【关键】注册反归一化参数 ---
        # 必须确保 args 中包含 y_mean 和 y_scale (Tensor)，用于 Ramp 约束计算
        if hasattr(args, 'y_mean') and hasattr(args, 'y_scale'):
            self.register_buffer('y_mean', args.y_mean)
            self.register_buffer('y_scale', args.y_scale)
            print("  [Config] Denormalization parameters loaded into Model.")
        else:
            self.register_buffer('y_mean', torch.tensor(0.0))
            self.register_buffer('y_scale', torch.tensor(1.0))
            print("  [Warning] y_mean/y_scale not found. Ensure they are passed in Phase 2.")

        # --- 辅助函数 ---
        def get_tensor(name, default=0.0):
            val = getattr(args, name, default)
            return val if isinstance(val, torch.Tensor) else torch.tensor(val).float()

        # 这些现在是真实的物理值 (MW)
        p_min = get_tensor('p_min')
        p_max = get_tensor('p_max')
        r_up = get_tensor('ramp_up')
        r_down = get_tensor('ramp_down')

        self.register_buffer('p_min', p_min)
        self.register_buffer('p_max', p_max)

        # --- 约束层 (基于物理值) ---
        self.cap_layer = CapacityConstraintLayer(p_min, p_max)
        self.ramp_layer = RampClamp(p_min, p_max, r_up, r_down)

        if hasattr(args, 'full_ptdf') and hasattr(args, 'full_limits'):
            self.trans_layer = FullNetworkTransmissionLayer(args.full_ptdf, args.full_limits)
            if hasattr(args, 'gen_map_index'):
                self.register_buffer('gen_map_index', args.gen_map_index.long())
            self.use_flow_check = True
            print(">>> Full Network Transmission Layer Initialized (Physical).")
        else:
            self.use_flow_check = False

        # --- 网络架构 ---
        self.encoder = SparsePhysicsEmbedding(args.input_size, self.num_nodes, input_mask)
        self.gcn1 = FastGCNLayer(self.adj)
        self.gcn2 = FastGCNLayer(self.adj)

        # 【核心组件替换】使用新的多尺度分解
        self.decompsition = MultiScaleDecomp(input_dim=self.num_nodes, scales=[6, 24, 48])

        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        self.future_encoder = nn.Linear(1, self.num_nodes)

        self.decoder = nn.Sequential(
            nn.Linear(self.num_nodes, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.output_size * 2)
        )
        self._init_weights()

    def _init_weights(self):
        last_layer = self.decoder[-1]
        nn.init.xavier_uniform_(last_layer.weight)
        if last_layer.bias is not None:
            nn.init.zeros_(last_layer.bias)
            last_layer.bias[self.output_size:].data.fill_(2.0)

    def forward(self, x, x_future):
        # x: [B, T, Features] (归一化输入)

        # 1-4. 神经网络运算 (在归一化空间进行)
        x_emb = self.encoder(x)
        x_node = x_emb.permute(0, 2, 1)
        x_node = self.gcn1(x_node)
        x_node = self.gcn2(x_node)

        # 使用新的分解组件
        s_init, t_init = self.decompsition(x_node)

        trend_part = self.Linear_Trend(t_init)
        seasonal_part = self.Linear_Seasonal(s_init)
        x_pred = (seasonal_part + trend_part).permute(0, 2, 1)

        x_combined = x_pred + self.future_encoder(x_future)
        raw_out = self.decoder(x_combined)
        power_logits, gate_logits = torch.split(raw_out, self.output_size, dim=2)

        # =======================================================
        # 5. 物理约束与输出 (转换到物理空间 MW)
        # =======================================================

        # (1) 容量约束 (输出物理值)
        power_static_phys = self.cap_layer(power_logits)

        # (2) 爬坡约束 (需要物理的初始状态)
        # x 是归一化的，取最后时刻 -> 反归一化 -> 得到物理 P_init (MW)
        # 这是为了配合 train.py 中的非归一化 Label
        p_last_norm = x[:, -1, -self.output_size:]
        p_last_phys = p_last_norm * self.y_scale + self.y_mean

        # 影子状态处理 (基于物理值判断)
        is_off = p_last_phys < (self.p_min * 0.9)  # 使用新代码中的 0.9 系数
        p_init_shadow = torch.where(is_off, self.p_min, p_last_phys)

        # 物理爬坡修正
        power_stream_phys = self.ramp_layer(power_static_phys, p_init_shadow)

        # (3) 门控
        gate_soft = torch.sigmoid(gate_logits)
        gate_hard = (gate_soft > 0.5).float()
        gate_out = gate_hard - gate_soft.detach() + gate_soft

        # 最终输出 (MW)
        pred_phys_final = power_stream_phys * gate_out

        # 返回物理值 (MW) 和门控状态
        return pred_phys_final, gate_out