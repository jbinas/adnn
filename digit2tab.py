#!/usr/bin/python

import os, sys, gzip, cPickle
import numpy as np


if __name__ == '__main__':

    n = int(sys.argv[1])
    f = gzip.open(sys.argv[-1], 'rb')
    train_set, valid_set, test_set = cPickle.load(f)

    dim = np.sqrt(len(test_set[0][0])).astype(int)
    x, y = np.meshgrid(range(dim), range(dim))

    dat = np.array(zip(x.flatten(), y.flatten(), test_set[0][n])).reshape((dim, dim, 3))

    with open('mnist-test-%s.dat' % sys.argv[1], 'w') as f:
        f.writelines('\n\n'.join(['\n'.join([' '.join([str(v) for v in p]) for p in col]) for col in dat]))
