#!/usr/bin/python

"""
Run measurements and/or training/testing

  Command line args:
      * demo - run demo on small network
      * measure - run measurements with specified parameters
      * train - run training using measured data (use together with
        measure or specify model parameters using load arg)
      * test - run subset of test set through circuit (use together
        with load)
      * rndseed=42 - specify numpy rng seed
      * dataset=data/datasets/<my-dataset>.pkz - specify which
        dataset to use
      * single=n - only run the nth measurement
      * load=data/measured/<my-measurement>.npy - load set of
        measurements to be used in training
      * params=data/devparams/<my-params>.pkl - device params
        to be used in simulation
      * numthreads=8 - number of threads to be used
      * N=196x49x10 - network dimensions
      * numtest=50 - number of test set samples to run
      * epochs=100 - number of training epochs
"""


import os
import sys
import time
import pickle
import numpy as np
from mp_tools import parmap
from build_nn import build_circuit, simulate, _ifactor
from scipy.optimize.minpack import curve_fit, leastsq
from pyNetlist import ParamList


if __name__ == '__main__':
    #default params
    run_training = False
    run_test = False
    run_demo = False
    run_measure = False
    NUM_THREADS = 8
    pattern_id = None
    xy_loaded = None
    params_loaded = None
    fractional_bits = 1
    integer_bits = 0
    RNDSEED = 42
    N = [196, 50, 10] # net dimensions
    dataset = 'mnist'
    num_test = 50 # number of test set samples to run
    num_inputs = 40 # number of measurements
    train_epochs = 100
    ifactor = _ifactor
    curve_scale = 1/1.75 #makes training independent of max. weight value
    lr = .005
    ds_scale = .1
    act_sparseness = 0.0001
    wgt_sparseness = 0.00001
    init_scale = 0.5
    visualize = True
    output_scale = 3.

    if 'train' in sys.argv:
        run_training = True
    if 'test' in sys.argv:
        run_test = True
    if 'measure' in sys.argv:
        run_measure = True
    if 'demo' in sys.argv:
        run_demo = True
    for argv in sys.argv:
        key, val = (argv.split('=') + [None])[:2]
        if key == 'rndseed':
            RNDSEED = int(val)
        if key == 'dataset':
            dataset = val
        if key == 'single':
            pattern_id = int(val)
        if key == 'load':
            xy_loaded = np.load(val)
        if key == 'params':
            params_loaded = pickle.load(open(val, 'r'))
        if key == 'numthreads':
            NUM_THREADS = int(val)
        if key == 'N':
            N = [int(i) for i in val.split('x')]
        if key == 'numtest':
            num_test = int(val)
        if key == 'visualize':
            visualize = int(val)
        if key == 'epochs':
            train_epochs = int(float(val))
        if key == 'lr':
            lr = float(val)
        if key == 'ds':
            ds_scale = float(val)
        if key == 'as':
            act_sparseness = float(val)
        if key == 'ws':
            wgt_sparseness = float(val)
        if key == 'is':
            init_scale = float(val)
        if key == 'os':
            output_scale = float(val)

    # --- all params set ---

    exec('from datasets import load_%s as load_data' % dataset)

    np.random.seed(RNDSEED)

    if run_training or run_test or run_demo:
        from keras import optimizers
        from keras.models import Sequential
        from keras.layers import Activation, BatchNormalization
        from keras.regularizers import l1, l2, activity_l1, activity_l2
        from keras_custom import ParameterizedLayer, ActivitySparseness, round_weights, act_sq_error, HardBounds, Inh_l1


def flatten_params(arr):
    """ flattens hierarchical parameter representation """
    out = []
    if isinstance(arr, ParamList):
        return arr.elements.tolist()
    elif isinstance(arr, dict):
        for k in arr.keys():
            out += flatten_params(arr[k])
    elif isinstance(arr, list):
        for v in arr:
            out += flatten_params(v)
    else:
        return [arr]
    return out

def get_io(N, inp, weights, params, **kwargs):
    """ sets off one simulation instance (used for multi-threading). """
    input = '%sn' % inp
    wgts_flat = np.concatenate([w.flatten() for w in weights])
    circuit, nodes, params = build_circuit(N,
            weights=weights,
            input=input,
            **params)
    return simulate(circuit,
            probe=nodes['probes'],
            ref=hash(tuple(wgts_flat) + (inp,))
            )

def build_weights(N, num_trials, layer_scale=None, layer_fanin=None):
    """ construct sparse, random, roughly isotropic connectivity matrices. """
    out = []
    fanin = layer_fanin or [1] * len(N)
    fixed_fanin = lambda n, dim: np.array([
        np.random.permutation(np.hstack((np.ones(n), np.zeros(dim[1]-n)))) \
                for _ in xrange(dim[0])])
    for t in xrange(num_trials):
        wgtset = []
        for l in xrange(len(N)-1):
            fanout_pre = np.zeros(N[l], dtype=int)
            for _wgtset in out:
                fanout_pre += _wgtset[l].sum(axis=0)
            w = np.zeros((N[l+1], N[l]), dtype=int)
            #try to distribute connections uniformly (only if fanin=1)
            for k in xrange(np.max(fanout_pre)+1):
                rid = np.random.permutation(np.where(np.all(w == 0, axis=1))[0])
                cid = np.random.permutation(np.where(
                    np.logical_and(np.all(w == 0, axis=0), fanout_pre == k))[0])
                dim = min(len(rid), len(cid))
                if dim and fanin[l] == 1:
                    w[[[i] for i in rid[:dim]], cid[:dim]] = np.eye(dim)
            #distribute remaining ones randomly
            rid = np.where(np.all(w == 0, axis=1))[0]
            if len(rid):
                w[rid, :] = fixed_fanin(fanin[l], (len(rid), w.shape[1]))
            wgtset.append(w)
        out.append(wgtset)
    if layer_scale is not None:
        return [[w.astype(float) * ls for w, ls in zip(ws, layer_scale)] for ws in out]
    return out

def measure(N, nocache=False, **kwargs):
    '''Simulate circuit of given dimensions and measure transfer curves'''
    inputs = kwargs.get('inputs', np.logspace(-2, 3, 6, base=2))
    layer_scale = kwargs.get('layer_scale', [1.75] * (len(N) - 1)) + [1.]
    extra_fanin = kwargs.get('layer_fanin', [])
    params_given = kwargs.get('params', {})

    #compile circuit once to hash parameters
    circuit, nodes, params = build_circuit(N, **params_given)
    params_flat = flatten_params(params)
    file_id = hash((RNDSEED,) + tuple(inputs) + tuple(layer_scale) + tuple(extra_fanin) + tuple(N) + tuple(params_flat))
    fname = 'data/measured/data_%s.npy' % file_id
    print 'looking for cached results: %s' % fname
    if os.path.isfile(fname) and not nocache:
        #results have been computed before -- use cached data
        print 'Using cached results..'
        try:
            outdata = np.load(fname)
            return outdata, params
        except IOError:
            print 'Error loading cached data -- recomputing...'

    weights = build_weights(N, len(inputs), layer_scale)
    argset = [{
        'N': N, 'inp': inp,
        'weights': wgtset,
        'params': params,
        'rec_layers': range(len(N)),
        } for inp, wgtset in zip(inputs, weights)]

    #extra input to higher layers
    for l in xrange(len(N)):
        lfiset = [[1] * l + [ef] + [1] * (len(N) - l) for ef in extra_fanin]
        for inp in inputs[-5:]:
            argset += [{
                'N': N, 'inp': inp,
                'weights': build_weights(N, 1, layer_scale, layer_fanin)[0],
                'params': params,
                'rec_layers': range(l + 1, len(N)),
                } for layer_fanin in lfiset]
    if pattern_id is not None:
        argset = [argset[pattern_id]]
    #run..
    data = parmap(get_io, argset, nprocs=NUM_THREADS)

    #sort data
    outdata = [[[] for j in xrange(n)] for n in N]
    #outdata contains a list of x-y pairs for every cell in every layer
    for sample, pars in zip(data, argset):
        wgtset_ = pars['weights'] + [np.eye(N[-1]).astype(int)] #output~ident.
        for l in pars['rec_layers']: #loop through layers
            fanout = wgtset_[l].sum(axis=0) / layer_scale[l]
            ids = np.where(fanout > 0)[0]
            x = sample[l][:, 1][ids]
            y = wgtset_[l].T.dot(sample[l+1][:, 1])[ids] / (fanout[ids]*layer_scale[l]**2)
            for k, id in enumerate(ids): #units that could be measured
                outdata[l][id].append((x[k], y[k]))
    fname = 'data/measured/data_%s%s.npy' % (file_id, '-'+str(pattern_id) if pattern_id is not None else '')
    np.save(fname, outdata)
    print 'measurement data written to %s' % fname
    fname = 'data/devparams/params_%s.pkl' % file_id
    pickle.dump(params, open('data/devparams/params_%s.pkl' % file_id, 'w'))
    print 'device parameters written to %s' % fname
    return outdata, params

def run_sample(N, params, weights, x, transient=False, **kwargs):
    ''' Run single input pattern through circuit '''
    input = ['%sn' % i for i in x]
    if transient:
        input = ['PULSE(0 %s)' % i for i in input]
    circuit, nodes, params = build_circuit(N,
            weights=weights, input=input, **params)
    mode = 'tran' if transient else 'op'
    probe = nodes['probes']
    return simulate(circuit, mode = mode, probe = probe, **kwargs)[-1][:, 1]

def fit_data(data, fct):
    ''' try to fit specified model to measured data '''
    f = lambda p, dat: fct(dat[:, 0], *p) - dat[:, 1] # objective fct
    num_params = fct.__code__.co_argcount - 1
    out = []
    for l in xrange(len(data)):
        out_l = []
        for i in xrange(len(data[l])):
            cell = np.array(data[l][i]) * 1e9
            gtz = cell[:, 1] > 0
            try:
                popt, pcov = curve_fit(fct, cell[gtz, 0], cell[gtz, 1])
            except:
                print 'automatic fitting failed. trying ls'
                try:
                    popt, pcov = leastsq(f, (1,) * num_params, args=(cell,))
                except:
                    raise Exception('Could not find parameters for cell %s-%s' % (l, i))
            out_l.append(popt)
        out.append(np.array(out_l))
    return out

def plot_points(data, fct=None, params=None, limits=(0, 10)):
    import matplotlib.pyplot as plt
    x = np.linspace(*limits)
    for i in xrange(len(data)):
        plt.subplot('1%s%s' % (len(data), i + 1))
        for j in xrange(len(data[i])):
            dat = np.array(data[i][j])
            color = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(dat[:, 0] * 1e9, dat[:, 1] * 1e9, 'o', c=color)
            if params is not None and fct is not None:
                plt.plot(x, fct(x, *params[i][j]), c=color)
        plt.grid()
        plt.title('Layer %s' % i)
        plt.xlabel('I')
    plt.show()



if __name__ == '__main__':
    """
    Run the full experiment
    """
    #input currents (in nA) to be used for measurements
    inputs = np.linspace(-2, 76, num=num_inputs)
    fit_fct = lambda x, a: a*x #model to be fitted to measured data

    if run_demo:
        import matplotlib.pyplot as plt
        #measure transfer curves for all layers of a small test net
        N_test = [20, 10, 5]
        xy_all, p_orig = measure(N_test, inputs=inputs, nocache=True)
        fit_params = fit_data(xy_all, fit_fct)
        plot_points(xy_all, fit_fct, fit_params, limits=(0, 42))
        raw_input("Showing measured data and fits. Press enter to build net")
        #build network
        pardict = [{'param_%s'%i:p[:, i] for i in xrange(p.shape[1])} for p in fit_params]
        model = Sequential()
        model.add(ParameterizedLayer(
            input_dim=N_test[0], output_dim=N_test[0],
            weights=[np.eye(N_test[0])], **pardict[0]))
        model.compile(loss='mse', optimizer='sgd')
        #run some inputs
        inp = np.array([np.ones(N_test[0]) * v for v in np.linspace(-5, 40)])
        output = model.predict(inp)
        plt.clf()
        plt.plot(inp[:, :N_test[0]], output, '-')
        plt.grid()
        plt.show()
        print "Finished test"

    if run_measure:
        xy_all, p_orig = measure(N, inputs=inputs)

    if run_training or run_test:
        #load/prep training data
        datasets = load_data(RNDSEED)
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        #output scale
        train_set_y *= output_scale
        valid_set_y *= output_scale

    if run_training:
        if xy_loaded is not None:
            xy_all = xy_loaded
        else:
            assert(run_measure)

        fit_params = fit_data(xy_all, fit_fct)
        #fit_params = [np.ones((n,1)) for n in N]
        fit_params = [p.T/p.mean() * curve_scale for p in fit_params]

        print "Preparing the data..."
        #build one-layer net to filter the input (to model input units)
        model = Sequential()
        model.add(ParameterizedLayer(
            input_dim=N[0], output_dim=N[0], weights=[np.eye(N[0])],
            integer_bits=1, fractional_bits=2, params_0=fit_params[0]))
        model.compile(loss='mse', optimizer='sgd')

        #filter input through input units and normalize
        #train_set_fil_x = model.predict(ds_scale * train_set_x)
        #valid_set_fil_x = model.predict(ds_scale * valid_set_x)
        #test_set_fil_x = model.predict( ds_scale * test_set_x)
        train_set_fil_x = model.predict(
            ds_scale * train_set_x / train_set_x.mean(axis=1, keepdims=True))
        valid_set_fil_x = model.predict(
            ds_scale * valid_set_x / valid_set_x.mean(axis=1, keepdims=True))
        test_set_fil_x = model.predict(
            ds_scale * test_set_x / test_set_x.mean(axis=1, keepdims=True))


        #build real net
        model = Sequential()
        #model.add(BatchNormalization(input_shape=(N[0],)))
        for l in xrange(len(N) - 1):
            model.add(ParameterizedLayer(
                input_dim=N[l], output_dim=N[l + 1], init_scale=(-init_scale,init_scale),
                integer_bits=integer_bits, fractional_bits=fractional_bits, ifactor=ifactor,
                params_0=fit_params[l+1].astype('float32'),
                #W_constraint=HardBounds(-ifactor*1.75,1.75),
                #activity_regularizer=ActivitySparseness(act_sparseness),
                activity_regularizer=activity_l1(act_sparseness) if l < len(N) - 2 and act_sparseness > 0 else None,
                W_regularizer=Inh_l1(wgt_sparseness) if l < len(N) - 2 and wgt_sparseness > 0 else None,
                ))
            #model.add(BatchNormalization())
        model.add(Activation('linear'))

        #init visualizer
        mon_layers = [l for l in model.layers[:-1]]
        if visualize:
            import matplotlib.pyplot as plt
            from visualizer import Visualizer
            visualizer = Visualizer(N, mon_layers, ifactor, integer_bits, fractional_bits)

        # Train a model
        print "Beginning training..."
        t1 = time.time()
        #optimizer = optimizers.SGD(lr=0.5, decay=1e-3, momentum=0.05, nesterov=False)
        optimizer = optimizers.Adam(lr=lr, epsilon=1e-6)
        model.compile(loss=act_sq_error, optimizer=optimizer, metrics=["accuracy"])
        log = model.fit(
                train_set_fil_x, train_set_y,
                nb_epoch=train_epochs, batch_size=200,
                validation_data=(valid_set_fil_x, valid_set_y),
                callbacks=[visualizer] if visualize else [],
                verbose=0,
                )
        loss, accuracy = model.evaluate(test_set_fil_x, test_set_y, batch_size=500, verbose=0)
        print "SW accuracy: " + str(accuracy * 100.)
        dt = (time.time() - t1) / 60
        print 'runtime:', dt, 'mins'

        #plot output histogram
        y_pred = model.predict(test_set_fil_x)
        if visualize:
            visualizer.save()
            plt.hist(y_pred.max(axis=1), bins=50, range=(0,2), alpha=.5)

        #Run SW model with rounded and clipped weights
        weights_raw = [l.get_weights() for l in mon_layers]
        weights_clipped = [round_weights(w[0], integer_bits, fractional_bits, ifactor) for w in weights_raw]
        fname = 'data/weights/weights_clipped.npy'
        np.save(fname, weights_clipped)
        print 'weights written to %s' % fname

        #plot output histogram
        y_pred = model.predict(test_set_fil_x)
        ids1 = np.where(y_pred.argmax(axis=1) == test_set_y.argmax(axis=1))[0]
        ids2 = np.where(y_pred.argmax(axis=1) != test_set_y.argmax(axis=1))[0]
        np.savetxt('data/tmp/y-pred.dat', y_pred)
        np.savetxt('data/tmp/y-pred-corr.dat', y_pred[ids1])
        np.savetxt('data/tmp/y-pred-err.dat', y_pred[ids2])
        if visualize:
            plt.hist(y_pred.max(axis=1), bins=50, range=(0,2), alpha=.5)
            plt.hist(y_pred[ids2].max(axis=1), bins=50, range=(0,2), alpha=.5)
            plt.show()
            raw_input("Press enter to quit.")

    if run_test:
        #Test in simulated circuit
        if not run_training:
            raise ValueError('Need to run training before testing')
        if params_loaded is not None:
            p_orig = params_loaded
        elif not run_measure:
            raise ValueError('Need to provide device params file or run measure first')
        print "Running subset of test set through circuit..."
        weights_clipped = [round_weights(w[0], integer_bits, fractional_bits, ifactor, prog_bits=True) for w in weights_raw]
        test_ids = np.random.choice(len(test_set_x), num_test, replace=False)
        i_in = [['%sn' % (i * 100) for i in test_set_x[j]] for j in test_ids]
        argset = [{'N': N,
                'params': p_orig,
                'weights': [w.T.astype(float) for w in weights_clipped],
                'ref': i,
                'x': pattern} for i, pattern in enumerate(i_in)]
        y_pred = parmap(run_sample, argset)
        y_true = test_set_y[test_ids]
        matches = [np.argmax(y_pred[i]) == np.argmax(y_true[i]) for i in xrange(num_test)]
        acc_circ = np.mean(matches)
        print "HW accuracy: " + str(acc_circ * 100.)

