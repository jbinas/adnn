#!/usr/bin/python

from build_nn import *
from mp_tools import parmap


def get_io(N, inp, weights, params, **kwargs):
    """ sets off one simulation instance (used for multi-threading). """
    wgts_flat = np.concatenate([w.flatten() for w in weights])
    circuit, nodes, params = build_circuit(N,
            weights=weights,
            input=inp,
            **params)
    return simulate(circuit,
            probe=nodes['probes'],
            ref=hash(tuple(wgts_flat) + (inp,))
            )



if __name__ == "__main__":
    N = [60] * 2
    inp = ['%sn' % v for v in [50, 100, 150, 200, 250, 300, 350, 400]]
    num_perm = len(inp)

    ibit = 0.25
    weights = [[
        np.vstack((
            #np.hstack((np.eye(N[0]/2), -0.5*np.eye(N[0]/2))),
            #np.hstack((np.eye(N[0]/2),  0.0*np.eye(N[0]/2)))
            np.hstack((np.ones((N[0]/2, N[0]/2)), -ibit * np.ones((N[0]/2, N[0]/2)))),
            np.hstack((np.ones((N[0]/2, N[0]/2)),   0.0 * np.ones((N[0]/2, N[0]/2))))
            ))] for _ in xrange(num_perm)]

    #get MC params
    circuit, nodes, params = build_circuit(N)
    argset = [{
        'N': N, 'inp': inpv,
        'weights': wgts,
        'params': params,
        } for inpv, wgts in zip(inp, weights)]
    data = parmap(get_io, argset)

    #sort data
    xy_all = [[[] for j in xrange(n)] for n in N]
    i = 0
    ylow = [[[] for _ in xrange(num_perm)] for n in N]
    yhgh = [[[] for _ in xrange(num_perm)] for n in N]
    for sample, wgtset in zip(data, weights):
        wgtset += [np.eye(N[-1]).astype(int)] #output~ident.
        for l in xrange(len(N)): #loop through layers
            y = sample[l + 1][:,1]
            for k in xrange(N[l]):
                xy_all[l][k].append((i, y[k]))
                if k < N[l]/2:
                    ylow[l][i].append(y[k])
                else:
                    yhgh[l][i].append(y[k])
        i += 1
    ylow_avg = np.array([[np.mean(a) for a in l] for l in ylow])
    ylow_std = np.array([[np.std(a) for a in l] for l in ylow])
    yhgh_avg = np.array([[np.mean(a) for a in l] for l in yhgh])
    yhgh_std = np.array([[np.std(a) for a in l] for l in yhgh])

    print (yhgh_avg - ylow_avg) / (yhgh_avg * ibit)
    print yhgh_std
    print ylow_std

    #do some plotting
    for i in xrange(len(N)):
        #plot output vs. actual input
        plt.figure()
        for j in xrange(N[i]):
            dat = np.array(xy_all[i][j])
            color = 'b' if j < N[i]/2 else 'r'
            plt.scatter(dat[:, 0] + 1, dat[:, 1] * 1e6, c=color)
        plt.grid()
        plt.title('Layer %s' % i)
        plt.xlabel('Iin (uA)')
        plt.ylabel('Iout (uA)')
    plt.show()

