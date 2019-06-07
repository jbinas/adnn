#!/usr/bin/python

"""
Run transient simulation

  Command line args:
      * rndseed=42 - numpy random seed
      * N=196x49x10 - network dimensions
      * inp=100 - maximum input current to be applied to input nodes (in
        nA)
      * inpat1=1 - input pattern 1 from dataset
      * inpat2=2 - input pattern 2 from dataset
      * weights=data/weights/<my-weights>.npy - set of weights to
        be used in simulation
      * params=data/devparams/<my-params>.pkl - device params to
        be used in simulation
      * dataset=data/datasets/<my-dataset>.pkz - specify which
        dataset to use
"""

import sys
import time
import numpy as np
from build_nn import build_circuit, simulate, _ifactor
from round_weights import round_weights


if __name__ == "__main__":

    #default params
    N = [196, 49, 10]
    inp = 100
    dataset = 'mnist'
    inpat1 = None
    inpat2 = None
    inpat3 = None
    weights = None
    params = None
    rndseed = 42
    ifactor = _ifactor

    for argv in sys.argv:
        key, val = (argv.split('=') + [None])[:2]
        if key == 'rndseed':
            rndseed = int(val)
        if key == 'N':
            N = [int(i) for i in val.split('x')]
        if key == 'inp':
            inp = int(val)
        if key == 'inpat1':
            inpat1 = int(val)
        if key == 'inpat2':
            inpat2 = int(val)
        if key == 'inpat3':
            inpat3 = int(val)
        if key == 'weights':
            weights = val
        if key == 'params':
            params = val
        if key == 'dataset':
            dataset = val

    # --- all params set ---
    exec('from datasets import load_%s as load_data' % dataset)

    np.random.seed(rndseed)

    if dataset is not None:
        datasets = load_data(rndseed)
        test_set_x, test_set_y = datasets[2]
        inpat1_ref = '%s~%s' % (inpat1, test_set_y[inpat1])
        inpat2_ref = '%s~%s' % (inpat2, test_set_y[inpat2])
        inpat3_ref = '%s~%s' % (inpat3, test_set_y[inpat3])
    else:
        inpat1_ref = inpat1
        inpat2_ref = inpat2
    #unique file reference
    ref = '%s-%s-%s-%s-%s-%s-%s' % (rndseed, 'x'.join([str(n) for n in N]), inp, inpat1_ref, inpat2_ref, inpat3_ref, time.time())

    #simulation duration
    tpat = 10
    tsim = tpat * 2 + 2

    #init params...
    if weights is not None:
        weights = np.load(weights)
        weights = [round_weights(w, 1, 2, ifactor, True).T.astype(float) for w in weights]
    else:
        weights = [np.random.normal(0, 3, (N[i + 1], N[i])).round().clip(-7, 7) for i in xrange(len(N)-1)]
    if params is not None:
        import pickle
        params = pickle.load(open(params, 'r'))
    else:
        circuit, nodes, params = build_circuit(N) #compile once to get params
    if inpat1 is not None:
        inp1 = test_set_x[inpat1]
        inp1 = 10 * inp * inp1 / inp1.sum()
    else:
        inp1 = np.random.rand(N[0]) * inp
    if inpat2 is not None:
        inp2 = test_set_x[inpat2]
        inp2 = 10 * inp * inp2 / inp2.sum()
    else:
        inp2 = np.random.rand(N[0]) * inp
    if inpat3 is not None:
        inp3 = test_set_x[inpat3]
        inp3 = 10 * inp * inp3 / inp3.sum()
    else:
        inp3 = np.random.rand(N[0]) * inp

    print 'Running %s network.' % ' x '.join([str(i) for i in N])
    tstart = time.time()
    circuit, nodes, params = build_circuit(N, weights=weights,
        input=['PWL(0 %sn 2us %sn 2us %sn 12us %sn 12us %sn)' % v for v in zip(inp1, inp1, inp2, inp2, inp3)], **params)
    data = simulate(circuit, mode='tran',
        probe=nodes['probes'] + [[nodes['vsupp']]],
        ref=ref, tstop='%sus'%tsim, tstep='10ns')
    #check whether this was classified OK
    st = data[-2][:, -1, 1] # final state
    if st.argmax() == test_set_y[inpat3]:
        print 'Classified correctly (%s)' % (st.argmax() + 1)
    else:
        print 'Misclassified (%s =/= %s)' % (st.argmax() + 1, test_set_y[inpat3] + 1)

    fname = 'data/analysis/data-%s.out.npy' % ref
    np.save(fname, [data, [weights, inp1, inp2, inp3]])
    rtime = (time.time() - tstart) / 60.
    print 'Runtime: %s min' % round(rtime, 2)
    print 'Data written to: %s' % fname

