import numpy as np
import torch
import argparse
from run_experiment import run_experiment
import random

parser = argparse.ArgumentParser("KOOPMAN AER")
parser.add_argument('--dataset', type=str, default='IEEEPPG', help='Dataset to use')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--grid-epochs', type=int, default=7, help='Number of epochs to train when doing grid search')
parser.add_argument('--normalize', type=bool, default=True, help='Whether to normalize the data')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

run_experiment(args.dataset, args.batch_size, args.epochs, args.device,
 l_rec=1, l_reg=1, l_koopman=1, koopman_size=120, should_normalize=args.normalize)

