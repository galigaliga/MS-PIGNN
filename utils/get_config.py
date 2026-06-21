import argparse


def get_config():
    parser = argparse.ArgumentParser(description='Power Prediction Framework')

    parser.add_argument('--data_path', type=str, default='./data', help='Data folder path')
    parser.add_argument('--graph_path', type=str, default='./ieee118new', help='Graph data path')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Model save path')

    parser.add_argument('--seq_len', type=int, default=144, help='Input history length')
    parser.add_argument('--pred_len', type=int, default=24, help='Prediction horizon')
    parser.add_argument('--step_size', type=int, default=1, help='Sliding window step size')
    parser.add_argument('--input_size', type=int, default=0, help='Auto-calculated in dataloader')
    parser.add_argument('--output_size', type=int, default=54)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--factor', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='MSPIGNN')


    # --- ALM (Augmented Lagrangian Method) 参数 ---
    # mu_0: 初始惩罚系数
    parser.add_argument('--mu_init', type=float, default=1, help='Initial penalty coefficient')
    # beta: 惩罚系数增长倍率
    parser.add_argument('--beta', type=float, default=1.1, help='Penalty growth factor')
    # max_mu: 防止 mu 无限增大导致数值爆炸
    parser.add_argument('--max_mu', type=float, default=100, help='Maximum penalty coefficient')

    # 模型参数
    args, unknown = parser.parse_known_args()
    if args.model_name == 'MSPIGNN':
        parser.add_argument('--node_hidden', type=int, default=128, help='Hidden dimension for GNN nodes')
        parser.add_argument('--moving_avg', type=int, default=25, help='Kernel size for Series Decomposition')

    args = parser.parse_args()
    return args