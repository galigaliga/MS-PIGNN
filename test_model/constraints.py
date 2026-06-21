import torch
import torch.nn as nn
import torch.nn.functional as F


class CapacityConstraintLayer(nn.Module):
    """
    [静态硬约束] 出力上下限约束层
    输入: Logits (无界)
    输出: 物理出力 MW (在 Pmin 到 Pmax 之间)
    """

    def __init__(self, p_min, p_max):
        super(CapacityConstraintLayer, self).__init__()
        self.register_buffer('p_min', p_min)
        self.register_buffer('p_max', p_max)

    def forward(self, logits):
        # Sigmoid [0,1] -> [Pmin, Pmax]
        return torch.sigmoid(logits) * (self.p_max - self.p_min) + self.p_min


class RampClamp(nn.Module):
    """
    [动态硬约束] 可微的爬坡修正层
    支持 118/2383 节点，包含物理启动逻辑
    """

    def __init__(self, p_min, p_max, r_up, r_down):
        super(RampClamp, self).__init__()
        self.register_buffer('p_min', p_min)
        self.register_buffer('p_max', p_max)
        self.register_buffer('r_up', r_up)
        self.register_buffer('r_down', r_down)

    def forward(self, pred_seq, p_init):
        # pred_seq: [Batch, Time, Units] (物理值)
        # p_init:   [Batch, Units]       (物理值)

        outputs = []
        prev_p = p_init

        # 定义关机阈值 (例如 Pmin 的 5%)
        # 用于判断上一时刻是否处于关机/极低出力状态
        off_threshold = self.p_min * 0.05

        # 显式循环时间步
        for t in range(pred_seq.shape[1]):
            raw_p = pred_seq[:, t, :]

            # 1. 启动逻辑判断
            # 如果 prev_p 接近 0，说明上一刻是关机
            is_startup = prev_p < off_threshold

            # 2. 计算动态上限
            # 正常爬坡: 上限 = prev + r_up
            normal_max = prev_p + self.r_up

            # 启动豁免: 如果是启动瞬间，允许直接跳变到 P_min
            # 取 max(P_min, normal_max) 确保即使 ramp_up 很小也能开机
            startup_max = torch.max(self.p_min, normal_max)

            # 根据状态选择上限
            dynamic_max = torch.where(is_startup, startup_max, normal_max)

            # 3. 计算动态下限
            dynamic_min = prev_p - self.r_down

            # 4. 与静态物理边界取交集
            effective_max = torch.min(self.p_max, dynamic_max)
            # 注意: 如果是启动，dynamic_min 可能是负数，P_min 会将其截断回 P_min (或0由Gate控制)
            effective_min = torch.max(self.p_min, dynamic_min)

            # 5. 硬截断
            corrected_p = torch.clamp(raw_p, min=effective_min, max=effective_max)

            outputs.append(corrected_p)
            prev_p = corrected_p

        return torch.stack(outputs, dim=1)


class FullNetworkTransmissionLayer(nn.Module):
    """
    [全局软约束] 全网直流潮流计算层
    使用 nn.Linear 存储 PTDF (冻结权重)
    """

    def __init__(self, full_ptdf_matrix, full_limits):
        super(FullNetworkTransmissionLayer, self).__init__()

        if not torch.is_tensor(full_ptdf_matrix):
            full_ptdf_matrix = torch.tensor(full_ptdf_matrix, dtype=torch.float32)

        num_lines, num_nodes = full_ptdf_matrix.shape

        # 使用 Linear 层存储矩阵乘法 (Bias=False, Weight=PTDF)
        # Linear: y = xA^T, 这里 A=PTDF
        self.projection = nn.Linear(num_nodes, num_lines, bias=False)

        with torch.no_grad():
            self.projection.weight.copy_(full_ptdf_matrix)
            self.projection.weight.requires_grad = False

        self.register_buffer('limits', full_limits)

    def forward(self, p_net_injection):
        """
        输入: 节点净注入功率 MW [Batch, Time, Nodes]
        输出: 平均越限值, 最大越限值
        """
        # Linear 层计算线路潮流: Flow = PTDF * Injection
        line_flows = self.projection(p_net_injection)

        # 计算越限 (ReLU保证只惩罚超过 Limit 的部分)
        # violations: [Batch, Time, Lines]
        violations = torch.relu(torch.abs(line_flows) - self.limits)

        return torch.mean(violations), torch.max(violations)