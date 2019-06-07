
import multiprocessing as mp
import errno

def get_queue(queue, block=True, timeout=None):
    ''' fix some buggy python-OS interaction '''
    while True:
        try:
            return queue.get(block, timeout)
        except IOError, e:
            if e.errno != errno.EINTR:
                raise

def join_proc(p):
    ''' fix some buggy python-OS interaction '''
    while True:
        try:
            return p.join()
        except IOError, e:
            if e.errno != errno.EINTR:
                raise

def fun(f,q_in,q_out):
    while True:
        i,x = get_queue(q_in)
        if i is None:
            break
        q_out.put((i,f(**x)))

def parmap(f, X, nprocs = mp.cpu_count()):
    q_in   = mp.Queue(1)
    q_out  = mp.Queue()
    proc = [mp.Process(target=fun,args=(f,q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()
    sent = [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]
    res = [get_queue(q_out) for _ in range(len(sent))]
    [join_proc(p) for p in proc]
    return [y for i,y in sorted(res)]
