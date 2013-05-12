from genome import Genome

import sys
import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
flags = mf.READ_WRITE | mf.COPY_HOST_PTR
sigs_a = np.ones(16, np.float32)
sigs_b = np.empty(16, np.float32)

def run(alpha, beta, delta, gamma, x0, y0):
    gstr = "+0{0}0{1}+0{2}1{3}-1{1}0{1}".format(alpha, beta, delta, gamma)
    g = Genome(gstr)
    sigs_a[0] = x0
    sigs_a[1] = y0
    sigs_a_buf = cl.Buffer(ctx, flags, hostbuf=sigs_a)
    sigs_b_buf = cl.Buffer(ctx, flags, hostbuf=sigs_b)
    program = cl.Program(ctx, g.export_cl()).build()
    trace_a = np.empty(50, np.float32)
    trace_b = np.empty(50, np.float32)
    for iteration in range(50):
        program.genome(queue, (1, 1), None, sigs_a_buf, sigs_b_buf)
        sigs_a_buf, sigs_b_buf = sigs_b_buf, sigs_a_buf
        if iteration == 48 or iteration == 49:
            cl.enqueue_copy(queue, sigs_b, sigs_b_buf).wait()
            trace_a[iteration] = sigs_b[0]
            trace_b[iteration] = sigs_b[1]
    d0 = abs(trace_a[-1] - trace_a[-2])
    d1 = abs(trace_b[-1] - trace_b[-2])
    if d0 > 0.01 and d1 > 0.01:
        return True
    else:
        return False

if __name__ == "__main__":
    for alpha in range(6, 10):
        for beta in range(1, 10):
            for delta in range(1, 10):
                for gamma in range(1, 10):
                    if run(alpha, beta, delta, gamma, 0.6, 0.3):
                        print(alpha, beta, delta, gamma, 0.6, 0.3)
        print("{0}{1}%".format(alpha, beta))
