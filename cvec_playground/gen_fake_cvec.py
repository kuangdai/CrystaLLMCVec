import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser(prog='gen_fake_cvec')
parser.add_argument('-d', '--dir', type=str, required=True,
                    help='data directory')
parser.add_argument('-n', '--n-cvec', type=int, default=2,
                    help='length of cvec')
args = parser.parse_args()

# find number of data from starts
with open(args.dir + '/starts_v1_train.pkl', 'rb') as fs:
    starts_train = pickle.load(fs)
with open(args.dir + '/starts_v1_val.pkl', 'rb') as fs:
    starts_val = pickle.load(fs)

# sample from N(0, 1)
cvec_train = np.random.randn(len(starts_train), args.n_cvec)
cvec_val = np.random.randn(len(starts_val), args.n_cvec)

# save
np.savez(args.dir + '/cvec_train', cvec_train)
np.savez(args.dir + '/cvec_val', cvec_val)
