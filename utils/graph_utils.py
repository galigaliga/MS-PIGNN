import torch
import pandas as pd
import numpy as np
import os


def load_graph_data(graph_path, num_nodes=118, device='cpu'):
    """
    [通用稀疏版] 读取 branch.csv 构建稀疏归一化邻接矩阵 (Sparse Tensor)
    适用于 IEEE 118 和 Polish 2383
    """
    branch_file = os.path.join(graph_path, 'branch.csv')
    if not os.path.exists(branch_file):
        raise FileNotFoundError(f"Branch file not found at {branch_file}")

    print(f"[GraphUtil] Building Sparse Topology from {branch_file}...")

    try:
        df = pd.read_csv(branch_file, header=0)
    except:
        df = pd.read_csv(branch_file, header=None)

    # 1. 提取边索引 (Source, Target)
    # 假设 Col 1=From, Col 2=To (0-based columns in dataframe)
    # IEEE 118 的 csv 通常也是这几列
    sources = df.iloc[:, 1].values.astype(int) - 1
    targets = df.iloc[:, 2].values.astype(int) - 1

    # 过滤越界节点
    mask = (sources >= 0) & (sources < num_nodes) & (targets >= 0) & (targets < num_nodes)
    sources = sources[mask]
    targets = targets[mask]

    # 2. 构建无向图索引 (双向边)
    # [修改] 先使用 np.array() 将列表转为单一 Numpy 数组，再转 Tensor，消除警告
    edge_index_np = np.array([
        np.concatenate([sources, targets]),
        np.concatenate([targets, sources])
    ])

    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)

    # 3. 添加自环 (Self-loops) -> A_hat = A + I
    self_loops = torch.arange(num_nodes, dtype=torch.long, device=device)
    self_loop_index = torch.stack([self_loops, self_loops], dim=0)

    edge_index = torch.cat([edge_index, self_loop_index], dim=1)

    # 4. 计算归一化系数 (GCN Normalization)
    # Val = 1 / sqrt(deg(row) * deg(col))

    # 计算度 (Degree)
    row, col = edge_index
    deg = torch.bincount(row, minlength=num_nodes).float()

    # D^-0.5
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # Norm values
    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # 5. 构建稀疏张量 [N, N]
    shape = torch.Size([num_nodes, num_nodes])

    # coalesce() 合并重复边并排序
    adj_sparse = torch.sparse_coo_tensor(edge_index, values, shape, device=device).coalesce()

    print(f"[GraphUtil] Sparse Adjacency Matrix Ready. Shape: {shape}, Edges: {adj_sparse._nnz()}")

    return adj_sparse, None


def build_input_mapping(data_path, graph_path, num_nodes=118, device='cpu'):
    """
    构建输入特征到图节点的物理映射矩阵
    动态检测机组数量，不再硬编码
    """
    print(f"[GraphUtil] Building Physics Mapping Matrix...")

    gen_file = os.path.join(graph_path, 'gen.csv')
    if not os.path.exists(gen_file):
        print("[Warning] gen.csv not found for mapping, using Identity.")
        return torch.eye(num_nodes, device=device)

    df_gen = pd.read_csv(gen_file)
    num_gens = len(df_gen)

    # Inputs: Load(N) + Gen(G)
    total_inputs = num_nodes + num_gens
    mapping = torch.zeros((num_nodes, total_inputs), device=device)

    # 1. Load -> Bus (Identity)
    indices = torch.arange(num_nodes, device=device)
    mapping[indices, indices] = 1.0

    # 2. Gen -> Bus
    try:
        # 假设第2列是 Bus ID
        gen_buses = df_gen.iloc[:, 1].values.astype(int) - 1

        gen_indices = torch.tensor(gen_buses, device=device, dtype=torch.long)
        col_offsets = torch.arange(num_gens, device=device, dtype=torch.long) + num_nodes

        valid_mask = (gen_indices >= 0) & (gen_indices < num_nodes)

        # 映射赋值
        mapping[gen_indices[valid_mask], col_offsets[valid_mask]] = 1.0

        print(f"    Mapped {valid_mask.sum().item()} Gen units.")

    except Exception as e:
        print(f"    [Warning] Failed to map Gen: {e}")

    return mapping