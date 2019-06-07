#!/usr/bin/python
"""
 Run analysis (plot/print output)

  Command line args:
      * noplot - do not plot output (used for batch processing)
      * noprint - do not print verbose output message (used for batch
        processing)

    $ python analysis.py data/analysis/<my-transient-data>.out.npy
"""

import sys
import numpy as np

if __name__ == "__main__":

    do_plot = True
    do_print = True
    vsupp = 1.8

    if len(sys.argv) < 2:
        raise IOError('Wrong number of arguments (%s required vs. %s given)' % (1, len(sys.argv) - 1))
    if 'noplot' in sys.argv:
        do_plot = False
    if 'noprint' in sys.argv:
        do_print = False
    if do_plot:
        import matplotlib.pyplot as plt
    fname = sys.argv[-1]
    try:
        args = fname.split('-')
        if len(args) == 7:
            prefix, rndseed, dim, inp, inpat1, inpat2, suffix = args
            inpat1, inpat1_res = [int(v) for v in inpat1.split('~')]
            inpat2, inpat2_res = [int(v) for v in inpat2.split('~')]
        else:
            prefix, rndseed, dim, inp, suffix = args
        N = [int(i) for i in dim.split('x')]
        data, [weights, inp1, inp2] = np.load(fname)
    except:
        raise IOError('File could not be loaded: %s' % fname)

    # compute winning unit
    st = data[-2][:, -1, 1] # final state
    tid = np.concatenate(([0], np.where(data[-2][:, :, 1].argmax(axis=0) != st.argmax())[0]))[-1] + 1
    tmax = data[-2][st.argmax(), tid, 0] # absolute convergence time
    first = (tid == 1) # convergence after 0s
    last = (tid == data[-1].shape[1] - 1) # no convergence
    valid = False if first or last else True
    clbit = 1 * (st.argmax() == inpat2_res)

    if do_plot:
        for l in xrange(1, len(N) + 2):
            if l == len(N) - 1: # skip input to last layer (show output instead)
                continue
            plt.figure()
            for i, d in enumerate(data[l]):
                lw = 2 if i == st.argmax() and l == len(N) else 1
                plt.plot(d[:-1, 0] * 1e6, d[:-1, 1] * 1e6, lw=lw)
                plt.scatter(d[[tid], 0] * 1e6, d[[tid], 1] * 1e6)
                plt.xlabel('time (us)')
                plt.ylabel('current (uA)')
                plt.title('Layer %s' % (l - 1 if l == len(N) else l))
                if l == len(N) + 1:
                    plt.title('Power')
    if valid:
        valid_times = data[-1][0, :-1, 0] <= tmax # (omit OP time)
        nops = np.array(N[1:]).dot(np.array(N[:-1])) # number of MACs
        times = data[-1][0, :-1][valid_times, 0]
        dt = times[1:] - times[:-1] #ignoring boundary
        en = -vsupp * data[-1][0, :-1][valid_times, 1][:-1].dot(dt)
        tops = nops / en / 1e12
    if do_print and valid:
        print 'Classified correctly' if clbit else 'Misclassified'
        print 'time to convergence: %s ns' % (tmax * 1e9)
        print 'throughput: %s GOpS' % (nops / tmax * 1e-9)
        print 'avg total power: %s uW' % (en / tmax * 1e6)
        print 'total energy: %s pJ' % (en * 1e12)
        print '%s TOp/J or %s pJ/Op' % (tops, 1 / tops)
    if do_plot:
        plt.show()
    elif valid and clbit:
        #returns: rndseed | din | dout | inp | convergence time | num ops | en | classified
        print '%s	%s	%s	%s	%s	%s	%s	%s	%s(%s)	%s(%s)	%s	%s' % (int(rndseed), N[0], N[-1], int(inp), tmax, nops, en, clbit, inpat1_res, inpat1, inpat2_res, inpat2, st.argmax(), st.max())
    else:
        if valid:
            conv = '1'
        else:
            conv = '0' if first else '-1'
        print '%% skipping sample (rndseed=%s, iin=%s, in1=%s, in2=%s, conv=%s, class=%s)' % (int(rndseed), int(inp), inpat1, inpat2, conv, clbit)

