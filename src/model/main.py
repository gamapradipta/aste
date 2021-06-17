import argparse
from model import ASTE

def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_len', type=int, default=100)    
    parser.add_argument('--exclude_padding', type=bool, default=True)
    parser.add_argument('--fine_tuned', type=bool, default=True)
    parser.add_argument('--bert_version', type=str, default='indobenchmark/indobert-base-p1')
    parser.add_argument('--bert_feature_dim', type=int, default=768)
    parser.add_argument('--nhop', type=int, default=1)
    parser.add_argument('--class_num', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    return parser

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    model = ASTE(args)
    model.init_model()