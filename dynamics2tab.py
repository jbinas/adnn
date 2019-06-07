#!/usr/bin/python

import sys
import numpy as np


if __name__ == '__main__':

    fname = sys.argv[-1]
    dat = np.load(fname)[0]
    t = dat[1][0,::20,0] - 1.99e-6
    t = t.reshape((t.shape[0],1))
    dout1 = np.hstack((t, dat[1][:,::20,1].T))
    dout2 = np.hstack((t, dat[2][:,::20,1].T))

    np.savetxt('dynamics_l1.dat', dout1 * 1e6)
    np.savetxt('dynamics_l2.dat', dout2 * 1e6)
