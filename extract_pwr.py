#!/usr/bin/python

import sys
import numpy as np


vsupp = 1.8
fname = sys.argv[-1]
verbose = False
stepsize = 1e-6
combine = True if 'combine' in sys.argv else False

if verbose:
    import matplotlib.pyplot as plt

if __name__ == "__main__":

    if combine:
        dat = np.loadtxt(fname)
        ids = np.arange(dat.shape[1])
        mean = dat.mean(axis=0)
        std = dat.std(axis=0)
        outdat = np.vstack((ids, mean, std, mean-std, mean+std))
        np.savetxt('data/analysis/pwr.dat', outdat.T)

    else:
        args = fname.split('-')
        prefix, rndseed, dim, inp, inpat1, inpat2, suffix = args
        inpat1, inpat1_res = [int(v) for v in inpat1.split('~')]
        N = [int(i) for i in dim.split('x')]
        data, [weights, inp1, inp2] = np.load(fname)

        nops = np.array(N[1:]).dot(np.array(N[:-1]))
        times = data[-1][0, :-1][:, 0]
        pwr = -vsupp * data[-1][0, :-1][:-1, 1]
        dt = times[1:] - times[:-1]
        #en = -vsupp * data[-1][0, :-1][:, 1][:-1].dot(dt)

        if verbose:
            print len(times), 'data points, tmax is', times[-1]
            plt.plot(times[:-1], pwr)
            plt.show()

        outdata = [0]
        steps = int(times[-1] / stepsize)
        for i in xrange(steps):
            valid = times[:-1] < (i + 1) * stepsize
            en = pwr[valid].dot(dt[valid])
            outdata.append(en)

        if verbose:
            plt.plot(outdata)
            plt.show()
        print ' '.join([str(v) for v in outdata])
