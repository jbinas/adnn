
from pyNetlist import *
from globalnets import *
from numpy import sqrt

_unit_weight = .25

class XNMOS(Device):
    name = 'XM'
    ports = ['d', 'g', 's', 'b']
    params = ['model', 'l', 'w', 'nf', 'AS', 'AD', 'PS', 'PD']

class XPMOS(Device):
    name = 'XM'
    ports = ['d', 'g', 's', 'b']
    params = ['model', 'l', 'w', 'nf', 'AS', 'AD', 'PS', 'PD']

#synapse circuit
syn = Circuit(name='syn',
        ports=['vinn', 'vinp', 'iout', 'w0', 'w1', 'w2', 'w3'],
        )
m00 = syn.addNode(XPMOS, d=vdd, b=vdd, g=syn.w0,
        model='pfet', l='.45u', w='.45u',
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u',
        )
m01 = syn.addNode(XNMOS, s=gnd, b=gnd, g=syn.w0,
        model='nfet', l='.45u', w='.45u',
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u',
        )
m02 = syn.addNode(XPMOS, d=m00.s, b=vdd, g=syn.vinp,
        model='pfet', l='.54u', w='.27u', nf=4,
        AS='0.24p', AD='0.24p', PS='1.3u', PD='1.3u',
        )
m03 = syn.addNode(XPMOS, d=m00.s, b=vdd, g=syn.vinp,
        model='pfet', l='.54u', w='.27u', nf=2,
        AS='0.24p', AD='0.24p', PS='1.3u', PD='1.3u',
        )
m04 = syn.addNode(XPMOS, d=m00.s, b=vdd, g=syn.vinp,
        model='pfet', l='.54u', w='.27u', nf=1,
        AS='0.24p', AD='0.24p', PS='1.3u', PD='1.3u',
        )
m05 = syn.addNode(XNMOS, s=m01.d, d=m02.s, b=gnd, g=syn.vinn,
        model='nfet', l='.54u', w='.27u', nf=4,
        AS='0.24p', AD='0.24p', PS='1.3u', PD='1.3u',
        )
m06 = syn.addNode(XNMOS, s=m01.d, d=m03.s, b=gnd, g=syn.vinn,
        model='nfet', l='.54u', w='.27u', nf=2,
        AS='0.24p', AD='0.24p', PS='1.3u', PD='1.3u',
        )
m07 = syn.addNode(XNMOS, s=m01.d, d=m04.s, b=gnd, g=syn.vinn,
        model='nfet', l='.54u', w='.27u', nf=1,
        AS='0.24p', AD='0.24p', PS='1.3u', PD='1.3u',
        )
m08 = syn.addNode(XNMOS, s=syn.iout, d=m02.s, b=gnd, g=syn.w1,
        model='nfet', l='.36u', w='.36u',
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u',
        )
m09 = syn.addNode(XNMOS, s=syn.iout, d=m03.s, b=gnd, g=syn.w2,
        model='nfet', l='.36u', w='.36u',
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u',
        )
m10 = syn.addNode(XNMOS, s=syn.iout, d=m04.s, b=gnd, g=syn.w3,
        model='nfet', l='.36u', w='.36u',
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u',
        )
syn.addNode(C, p1=syn.iout, p2=gnd, c='3.5f') #iout-gnd (accounting for memory part as well)
syn.addNode(C, p1=m02.s, p2=gnd, c='1.5f')  #w1-gnd
syn.addNode(C, p1=m03.s, p2=gnd, c='1f')    #w2-gnd
syn.addNode(C, p1=m04.s, p2=gnd, c='1f')    #w3-gnd
syn.addNode(C, p1=syn.vinn, p2=gnd, c='2f') #vn-gnd
syn.addNode(C, p1=syn.vinp, p2=gnd, c='2f') #vp-gnd


#soma circuit
som = Circuit(name='som',
        ports=['iin'],
        )
r1 = som.addNode(R, p1=som.iin, r='10')
m0 = som.addNode(XNMOS, s=gnd, b=gnd, g=r1.p2, d=r1.p2,
        model='nfet', l='.45u', w='2.7u',
        AS='1.45p', AD='1.45p', PS='6.5u', PD='6.5u')
som.addNode(C, p1=som.iin, p2=gnd, c='2f')

#axon circuit
axon = Circuit(name='axon',
        ports=['vin', 'voutn', 'voutp'],
        )
ri = axon.addNode(R, p1=axon.vin, r='10')
rn = axon.addNode(R, p1=axon.voutn, r='10')
rp = axon.addNode(R, p1=axon.voutp, r='10')
m0 = axon.addNode(XNMOS, d=rp.p2, g=ri.p2, s=gnd, b=gnd,
        model='nfet', l='.45u', w='2.7u',
        AS='1.45p', AD='1.45p', PS='6.5u', PD='6.5u',
        )
m1 = axon.addNode(XPMOS, d=vdd, g=rp.p2, s=rp.p2, b=vdd,
        model='pfet', l='.45u', w='2.7u',
        AS='1.45p', AD='1.45p', PS='6.5u', PD='6.5u',
        )
m2 = axon.addNode(XPMOS, d=vdd, g=rp.p2, s=rn.p2, b=vdd,
        model='pfet', l='.45u', w='2.7u',
        AS='1.45p', AD='1.45p', PS='6.5u', PD='6.5u',
        )
m3 = axon.addNode(XNMOS, d=rn.p2, g=rn.p2, s=gnd, b=gnd,
        model='nfet', l='.45u', w='2.7u',
        AS='1.45p', AD='1.45p', PS='6.5u', PD='6.5u',
        )

#out sink
sink = Circuit(name='sink',
        ports=['vin', 'iin'],
        )
rv = sink.addNode(R, p1=sink.vin, r='10')
ri = sink.addNode(R, p1=sink.iin, r='10')
m0 = sink.addNode(XNMOS, s=gnd, b=gnd, g=rv.p2, d=ri.p2,
        model='nfet', l='.45u', w='.45u',
        AS='0.24p', AD='0.24p', PS='1.5u', PD='1.5u')


#weight to binary conversion
low_elem, high_elem = gnd, vdd
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
        conf = {k : low_elem for k in keys}
        if w < 0:
            conf['w0'] = high_elem
        for i,bit in enumerate(bin(abs(int(w/_unit_weight)))[::-1]):
            if bit == 'b':
                break
            if bit == '1':
                conf['w%s' % (num_bits-i)] = high_elem
        return conf

