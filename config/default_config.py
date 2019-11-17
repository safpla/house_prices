import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--cross_valid_k', type=int, default=10)
    parser.add_argument('--dnn_lr', type=float, default=0.05)
    parser.add_argument('--neighborhood', type=bool, default=False)
    parser.add_argument('--dim_features', type=int, default=52)
    parser.add_argument('--dnn_valid_freq', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_iterations', type=int, default=20000)

    return parser.parse_args()
