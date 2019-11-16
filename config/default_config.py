import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--cross_valid_k', type=int, default=10)
    parser.add_argument('--dnn_lr', type=float, default=0.001)
    parser.add_argument('--neighborhood', type=bool, default=False)
    parser.add_argument('--dim_features', type=int, default=55)

    return parser.parse_args()
