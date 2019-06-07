
from pyNetlist import *
from globalnets import *
from numpy import sqrt

_unit_weight = .25
_ifactor = 2.3


#synapse circuit
syn = Circuit(name='syn',
        ports=['vinn', 'vinp', 'iout', 'w0','w1','w2','w3'],
        params=['d0','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10'],
        d0=0, d1=0, d2=0, d3=0, d4=0, d5=0, d6=0, d7=0, d8=0, d9=0, d10=0
        )
m00 = syn.addNode(PMOS, d=vdd, b=vdd, g=syn.w0,
        model='tsmc180p', delvto=syn.d0, l='.45u', w='.45u',
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u')
m01 = syn.addNode(NMOS, s=gnd, b=gnd, g=syn.w0,
        model='tsmc180n', delvto=syn.d1, l='.45u', w='.45u',
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u')
m02 = syn.addNode(PMOS, d=m00.s, b=vdd, g=syn.vinp,
        model='tsmc180p', delvto=syn.d2, l='.54u', w='.27u', m=4,
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u')
m03 = syn.addNode(PMOS, d=m00.s, b=vdd, g=syn.vinp,
        model='tsmc180p', delvto=syn.d3, l='.54u', w='.27u', m=2,
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u')
m04 = syn.addNode(PMOS, d=m00.s, b=vdd, g=syn.vinp,
        model='tsmc180p', delvto=syn.d4, l='.54u', w='.27u', m=1,
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u')
m05 = syn.addNode(NMOS, s=m01.d, d=m02.s, b=gnd, g=syn.vinn,
        model='tsmc180n', delvto=syn.d5, l='.54u', w='.27u', m=4,
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u')
m06 = syn.addNode(NMOS, s=m01.d, d=m03.s, b=gnd, g=syn.vinn,
        model='tsmc180n', delvto=syn.d6, l='.54u', w='.27u', m=2,
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u')
m07 = syn.addNode(NMOS, s=m01.d, d=m04.s, b=gnd, g=syn.vinn,
        model='tsmc180n', delvto=syn.d7, l='.54u', w='.27u', m=1,
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u')
m08 = syn.addNode(NMOS, s=syn.iout, d=m02.s, b=gnd, g=syn.w1,
        model='tsmc180n', delvto=syn.d8, l='.45u', w='.45u',
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u')
m09 = syn.addNode(NMOS, s=syn.iout, d=m03.s, b=gnd, g=syn.w2,
        model='tsmc180n', delvto=syn.d9, l='.45u', w='.45u',
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u')
m10 = syn.addNode(NMOS, s=syn.iout, d=m04.s, b=gnd, g=syn.w3,
        model='tsmc180n', delvto=syn.d10, l='.45u', w='.45u',
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u')
r1 = syn.addNode(R, p1=gnd, p2=m02.s, r='10g')
r2 = syn.addNode(R, p1=gnd, p2=m03.s, r='10g')
r4 = syn.addNode(R, p1=gnd, p2=m04.s, r='10g')
rn = syn.addNode(R, p1=gnd, p2=syn.vinn, r='10g')
rp = syn.addNode(R, p1=vdd, p2=syn.vinp, r='10g')
r00 = syn.addNode(R, p1=gnd, p2=m00.s, r='10g')
r01 = syn.addNode(R, p1=gnd, p2=m01.d, r='10g')
cn = syn.addNode(C, p1=gnd, p2=syn.vinn, c='2f')
cp = syn.addNode(C, p1=vdd, p2=syn.vinp, c='2f')
co = syn.addNode(C, p1=gnd, p2=syn.iout, c='3.5f')
syn.addNode(C, p1=m02.s, p2=gnd, c='1.5f')  #w1-gnd
syn.addNode(C, p1=m03.s, p2=gnd, c='1f')    #w2-gnd
syn.addNode(C, p1=m04.s, p2=gnd, c='1f')    #w3-gnd
syn_nodes = [m00, m01, m02, m03, m04, m05, m06, m07, m08, m09, m10]

#soma circuit
som = Circuit(name='som',
        ports=['iin'],
        params=['d0'],
        d0=0
        )
r1 = som.addNode(R, p1=som.iin, r='10')
m0 = som.addNode(NMOS, s=gnd, b=gnd, g=r1.p2, d=r1.p2,
        model='tsmc180n', delvto=som.d0, l='.45u', w='2.7u',
        AS='1.45p', AD='1.45p', PS='6.5u', PD='6.5u')
r1 = som.addNode(R, p1=gnd, p2=som.iin, r='10g')
som_nodes = [m0]

#axon circuit
axon = Circuit(name='axon',
        ports=['vin', 'voutn', 'voutp'],
        params=['d0', 'd1', 'd2', 'd3'],
        d0=0, d1=0, d2=0, d3=0
        )
ri = axon.addNode(R, p1=axon.vin, r='10')
rn = axon.addNode(R, p1=axon.voutn, r='10')
rp = axon.addNode(R, p1=axon.voutp, r='10')
m0 = axon.addNode(NMOS, d=rp.p2, g=ri.p2, s=gnd, b=gnd,
        model='tsmc180n', delvto=axon.d0, l='.45u', w='2.7u',
        AS='1.45p', AD='1.45p', PS='6.5u', PD='6.5u')
m1 = axon.addNode(PMOS, d=vdd, g=rp.p2, s=rp.p2, b=vdd,
        model='tsmc180p', delvto=axon.d1, l='.45u', w='2.7u',
        AS='1.45p', AD='1.45p', PS='6.5u', PD='6.5u')
m2 = axon.addNode(PMOS, d=vdd, g=rp.p2, s=rn.p2, b=vdd,
        model='tsmc180p', delvto=axon.d2, l='.45u', w='2.7u',
        AS='1.45p', AD='1.45p', PS='6.5u', PD='6.5u')
m3 = axon.addNode(NMOS, d=rn.p2, g=rn.p2, s=gnd, b=gnd,
        model='tsmc180n', delvto=axon.d3, l='.45u', w='2.7u',
        AS='1.45p', AD='1.45p', PS='6.5u', PD='6.5u')
axon_nodes = [m0,m1,m2,m3]

#out sink
sink = Circuit(name='sink',
        ports=['vin', 'iin'],
        params=['d0'],
        d0=0
        )
rv = sink.addNode(R, p1=sink.vin, r='10')
ri = sink.addNode(R, p1=sink.iin, r='10')
m0 = sink.addNode(NMOS, s=gnd, b=gnd, g=rv.p2, d=ri.p2,
        model='tsmc180n', delvto=sink.d0, l='.45u', w='.45u',
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u')
sink_nodes = [m0]


#compute vt offset variance
dvt = lambda n: 3.3e-3 / sqrt(float(n.l.value[:-1]) * float(n.w.value[:-1]) * (n.m.value if n.m.value is not None else 1))
syn_dvt = [dvt(n) for n in syn_nodes]
axon_dvt = [dvt(n) for n in axon_nodes]
som_dvt = [dvt(n) for n in som_nodes]
sink_dvt = [dvt(n) for n in sink_nodes]



#weight to binary conversion
def w2b(w, num_bits=3):
    keys = ['w%s' % i for i in xrange(num_bits+1)]
    if not isinstance(w, (int,float)) and len(w) > 1:
        conf = {k : [] for k in keys}
        for wi in w:
            confi = w2b(wi)
            for k in keys:
                conf[k].append(confi[k])
        return {k: PortList(conf[k]) for k in keys}
    else:
        if abs(w) > _unit_weight * 2**num_bits:
            raise ValueError('Weight value too large!')
        if w % _unit_weight != 0:
            raise ValueError('Weight values must be multiples of %s (provided value was %s).' % (_unit_weight,w))
        conf = {k : gnd for k in keys}
        if w < 0:
            conf['w0'] = vdd
        for i,bit in enumerate(bin(abs(int(w/_unit_weight)))[::-1]):
            if bit == 'b':
                break
            if bit == '1':
                conf['w%s' % (num_bits-i)] = vdd
        return conf

