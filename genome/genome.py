from collections import namedtuple
from string import Template

genome_cl = """
#define GRID_SIZE  (512)
#define GRID_CELLS (GRID_SIZE * GRID_SIZE)

// A slightly leaky activation function that saturates at about x=1.0
#define activation(x) native_divide(1.0f, 1.0f + native_exp(4.0f - 10.0f * x))

__kernel void genome(__global float16* sigs_in, __global float16* sigs_out)
{
    uint col = get_global_id(0);
    uint row = get_global_id(1);
    uint position = row*GRID_SIZE + col;

    __local float16 sigs = sigs_in[position];

    $degregation

    $production

    sigs_out[position] = sigs;
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
        return "sigs.s{sig} *= {factor};".format(sig=signal, factor=factor)

    def prodcode(self, gene):
        if gene.rbs == 0:
            return ""
        act = "activation(sigs.s{sig})".format(sig=gene.sig_in)
        if gene.reg == '-':
            act = "(1.0 - {act})".format(act=act)
        amt = "{act} * {rbs}".format(act=act, rbs=0.1*gene.rbs)
        return "sigs.s{out} += {amt};".format(out=gene.sig_out, amt=amt)
    
    def export_cl(self):
        degregation = "\n    ".join([self.degcode(s) for s in self.signals])
        production = "\n    ".join([self.prodcode(g) for g in self.genes])
        progstr = Template(genome_cl).substitute(
            degregation=degregation, production=production)
        return progstr

if __name__ == "__main__":
    g = Genome("+0504+0511-1804")
    print(g.export_cl())
