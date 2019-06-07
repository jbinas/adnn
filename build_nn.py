
import os
from pyNetlist import *
from pyNetlist.interfaces import spice
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import numpy as np
import matplotlib.pyplot as plt
from globalnets import vdd, gnd
from subcircuits import syn, som, axon, sink, syn_dvt, axon_dvt, som_dvt, sink_dvt, w2b, _ifactor


def build_circuit(N, **kwargs):
    num_fets = 5*sum(N)
    for l in xrange(len(N)-1):
        num_fets += 11*N[l]*N[l+1]
    print 'Generating circuit (~%s FETs)...' % num_fets

    #parse kwargs
    num_layers = len(N)
    if kwargs.get('verbose'):
        print ' - Initializing parameters...'
    #generate normal-distributed mismatch
    mm_som = kwargs.get('mm_som', [
                {'d%s'%i: ParamList(np.random.normal(0, som_dvt[i], N[l]))
                    for i in xrange(1)} for l in xrange(num_layers)])
    mm_sink = kwargs.get('mm_sink',
                {'d%s'%i: ParamList(np.random.normal(0, sink_dvt[i], N[-1]))
                    for i in xrange(1)})
    mm_axon = kwargs.get('mm_axon', [
                {'d%s'%i: ParamList(np.random.normal(0, axon_dvt[i], N[l]))
                    for i in xrange(4)} for l in xrange(num_layers)])
    mm_syn = kwargs.get('mm_syn', [
                [{'d%s'%i: ParamList(np.random.normal(0, syn_dvt[i], N[l]))
                    for i in xrange(11)} for j in xrange(N[l+1])]
                    for l in xrange(num_layers-1)] +
                [{'d%s'%i: ParamList(np.random.normal(0, syn_dvt[i], N[-1]))
                    for i in xrange(11)}])
    i_in_val = kwargs.get('input', '-1n')
    if isinstance(i_in_val, (list, np.ndarray)):
        i_in_val = ParamList(i_in_val)

    c = Circuit()
    vsupp = c.addNode(V, p1=vdd, p2=gnd, v=1.8) #set vdd
    i_in = c.addArray(I, p1=vdd, i=i_in_val, size=N[0]) #input current sources

    somata = []
    axons = []
    synaps = []
    probes = []
    for layer in xrange(num_layers):
        if kwargs.get('verbose'):
            print ' - Building layer %s...' % layer
        #somata
        if layer == 0: #connect to input
            probes.append(c.addArray(V, p1=i_in.p2, size=N[layer], v=0))
        else:
            probes.append(c.addArray(V, size=N[layer], v=0))
        somata.append(c.addArray(som.Instance,
                iin=probes[layer].p2, size=N[layer], **mm_som[layer]))
        if layer < num_layers-1:
            axons.append(c.addArray(axon.Instance,
                vin=probes[layer].p2, size=N[layer], **mm_axon[layer]))
        if layer > 0:
            #synapses
            if kwargs.get('weights') is not None:
                weights = [w2b(row) for row in kwargs.get('weights')[layer-1]]
            else: #set all weights to 1
                weights = [w2b([1]*N[layer-1]) for j in xrange(N[layer])]
            syn_arr = [c.addArray(syn.Instance, size=N[layer-1],
                    vinn=axons[layer-1].voutn, vinp=axons[layer-1].voutp,
                    iout=probes[layer].p1[j],
                    **dict(weights[j].items() + mm_syn[layer-1][j].items()))
                    for j in xrange(N[layer])]
            synaps.append(syn_arr)
    #output sinks
    synaps.append(c.addArray(sink.Instance, size=N[layer],
            vin=somata[layer].iin, **mm_sink))
    #output probes
    probes.append(c.addArray(V, p1=vdd, p2=synaps[-1].iin, v=0, size=N[layer]))
    return c, {
                'input': i_in,
                'somata': somata,
                'axons': axons,
                'synapses': synaps,
                'probes': probes,
                'vsupp': vsupp,
            }, {
                'mm_som': mm_som,
                'mm_axon': mm_axon,
                'mm_syn': mm_syn,
                'mm_sink': mm_sink,
            }

def simulate(circuit, probe, ref, mode='op', limits=None, stepsize=None, verbose=False, **kwargs):
    outfiles = []
    if isinstance(probe, Device):
        probe = [probe]
    #generate spice code
    f = File('data/spice/multilayer.cir')
    f.append(spice.comment('Multilayer NN'))
    f.append(spice.include('lib/tsmc180nmcmos_2.lib'))
    f.append(spice.command('global', 'VDD'))
    f.append(spice.netlist(circuit))
    f.append(spice.command('control'))
    f.append(spice.command('option', 'gmin=1e-11')[1:])
    f.append(spice.command('option', 'abstol=100p')[1:])
    f.append(spice.command('set', 'num_threads=2')[1:])
    if mode == 'tran':
        f.append(spice.command(mode,
            kwargs.get('tstep', '10ns'),
            kwargs.get('tstop', '1us'))[1:])
    else:
        #do operating point analysis
        f.append(spice.command(mode)[1:])
    if isinstance(probe[0], Device):
        probe = [probe]
    for l, pnodes in enumerate(probe):
        outfiles_l = []
        for pnode in pnodes:
            fname = 'data/spice/ngspice-out-%s-l%s-%s-%s' % (mode, l, pnode.ref, ref)
            f.append(spice.wrdata(fname.lower(), 'i', pnode))
            outfiles_l.append(fname.lower()+'.data')
        outfiles.append(outfiles_l)
    f.append(spice.command('endc'))
    f.append(spice.command('end'))
    f.write() #save file for reference
    #done generating spice code -- run simulation
    logfname = 'data/spice/ngspice-%s.log'%ref
    print 'Simulating circuit... (logfile: %s)' % logfname
    p = Popen(['ngspice', '-b', '-n', '-o', logfname],
            stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    stdout, stderr = p.communicate(input=str(f))
    if stderr or 'Error' in open(logfname).read():
        print stdout
        raise CalledProcessError(p.returncode, 'ngspice')
    if verbose:
        print stdout
    #read back generated data
    out = np.array([np.array([np.loadtxt(f) for f in f_l]) for f_l in outfiles])
    #clean up
    for f_l in outfiles:
        for f in f_l:
            os.remove(f)
    os.remove(logfname)
    return out


if __name__ == "__main__":
    N = [10, 11, 12]
    inp = 100
    layer_w = [1.75, 1.75, 1]

    weights = [np.eye(N[i + 1], N[i], dtype=int) * layer_w[i] for i in xrange(len(N) - 1)]
    circuit, nodes, params = build_circuit(N) #compile once to get params

    data = []
    for inp in inputs:
        circuit, nodes, params = build_circuit(N,
                weights=weights,
                input='%sn' % inp,
                **params)
        data.append(simulate(circuit,
                probe=nodes['probes'],
                ref=inp,
                ))

    #sort data
    xy_all = [[[] for j in xrange(n)] for n in N]
    for sample in data:
        wgtset = weights + [np.eye(N[-1]).astype(int)] #output~ident.
        for l in xrange(len(N)): #loop through layers
            fanout = np.dot(wgtset[l].T, np.ones((N + [N[-1]])[l + 1])) / layer_w[l]
            ids = np.where(fanout > 0)[0]
            x = sample[l][:, 1][ids]
            y = np.dot(wgtset[l].T, sample[l + 1][:, 1])[ids] / (fanout[ids] * layer_w[l] ** 2)
            for k, id in enumerate(ids): #units that could be measured
                xy_all[l][id].append((x[k], y[k]))

    #do some plotting
    for i in xrange(len(N)):
        #plot output vs. actual input
        plt.figure()
        for j in xrange(N[i]):
            dat = np.array(xy_all[i][j])
            color = [(np.random.random(), np.random.random(), np.random.random())]
            plt.scatter(dat[:, 0] * 1e9, dat[:, 1] * 1e9, c=color)
        plt.grid()
        plt.title('Layer %s' % i)
        plt.xlabel('I')
    plt.show()

