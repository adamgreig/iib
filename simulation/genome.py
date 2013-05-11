from collections import namedtuple
from string import Template

genome_cl = """
#define GRID_SIZE  (512)
#define GRID_CELLS (GRID_SIZE * GRID_SIZE)

// A slightly leaky activation function that saturates at about x=1.0
#define activation(x) native_divide(1.0f, 1.0f + native_exp(4.0f - 10.0f * x))

__kernel void genome(__global float* sigs_in, __global float* sigs_out)
{
    uint col = get_global_id(0);
    uint row = get_global_id(1);
    uint position = row*GRID_SIZE + col;

    __local float16 sigs;
    sigs = vload16(position, sigs_in);

    $degregation

    $production

    vstore16(sigs, position, sigs_out);
}
"""

Gene = namedtuple('Gene', 'reg sig_in rbs sig_out deg')

class Genome:
    signals = [str(i) for i in range(10)] + [chr(c) for c in range(65, 71)]

    def __init__(self, genes):
        self.genestr = genes
        self.parsegenes(genes)

    def parsegenes(self, genestr):
        self.genes = []
        for gene in [genestr[i:i+5] for i in range(0, len(genestr), 5)]:
            self.genes.append(Gene(gene[0], gene[1], int(gene[2]), gene[3],
                               int(gene[4])))

    def __str__(self):
        return self.genestr

    def degcode(self, signal):
        rbs_deg = [(g.rbs, g.deg) for g in self.genes if g.sig_out == signal]
        rbs_total = float(sum(w[0] for w in rbs_deg))
        deg_avg = sum((w[0]/rbs_total) * w[1] for w in rbs_deg) / 10.0
        if deg_avg == 0.0:
            return ""
        factor = 1.0 - deg_avg
        return "sigs.s{sig} *= {factor}f;".format(sig=signal, factor=factor)

    def prodcode(self, gene):
        if gene.rbs == 0:
            return ""
        act = "activation(sigs.s{sig})".format(sig=gene.sig_in)
        if gene.reg == '-':
            act = "(1.0f - {act})".format(act=act)
        amt = "{act} * {rbs}f".format(act=act, rbs=0.1*gene.rbs)
        return "sigs.s{out} += {amt};".format(out=gene.sig_out, amt=amt)
    
    def export_cl(self):
        degregation = "\n    ".join([self.degcode(s) for s in self.signals])
        production = "\n    ".join([self.prodcode(g) for g in self.genes])
        progstr = Template(genome_cl).substitute(
            degregation=degregation, production=production)
        return progstr

if __name__ == "__main__":
    gstr = "+0504+0511-1804"
    g = Genome(gstr)
    print("OpenCL code for 'genome {0}':".format(gstr))
    print(g.export_cl())
    print()
    import numpy as np
    import pyopencl as cl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    sigs_a = np.ones(512*512*16, np.float32) * 0.1
    sigs_b = np.zeros(512*512*16, np.float32)
    flags = mf.READ_WRITE | mf.COPY_HOST_PTR
    sigs_a_buf = cl.Buffer(ctx, flags, hostbuf=sigs_a)
    sigs_b_buf = cl.Buffer(ctx, flags, hostbuf=sigs_b)
    program = cl.Program(ctx, g.export_cl()).build()
    trace_a = np.empty(40, np.float32)
    trace_b = np.empty(40, np.float32)
    for iteration in range(40):
        program.genome(queue, (512, 512), None, sigs_a_buf, sigs_b_buf)
        sigs_a_buf, sigs_b_buf = sigs_b_buf, sigs_a_buf
        if iteration % 1 == 0:
            cl.enqueue_copy(queue, sigs_b, sigs_b_buf).wait()
            x = y = 0
            position = (y*512 + x)*16
            signals = ", ".join(str(s) for s in sigs_b[position:position+16])
            print("({0},{1}): [{2}]".format(x, y, signals))
            trace_a[iteration] = sigs_b[position]
            trace_b[iteration] = sigs_b[position+1]
    import matplotlib.pyplot as plt
    plt.plot(trace_a)
    plt.plot(trace_b)
    plt.show()
