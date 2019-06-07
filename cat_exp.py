#!/usr/bin/python

from run_exp import *

alldata = [np.load('data/measured/'+f) for f in os.listdir('data/measured') if f.endswith('npy')]

N = [len(arr) for arr in alldata[0]]

outdata = [[[] for j in xrange(n)] for n in N]

for sample in alldata:
    for l, sample_l in enumerate(sample): #loop through layers
        for i, sample_j in enumerate(sample_l): #units that could be measured
            if sample_j:
                outdata[l][i].extend(sample_j)


np.save('data/measured/data_cat', outdata)
