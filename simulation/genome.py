from collections import namedtuple
from string import Template

genome_cl_str = """//CL//
// A slightly leaky activation function that saturates at about x=1.0
#define activation(x) native_recip(1.0f + native_exp(4.0f - 10.0f * x))
#define POS (get_global_id(1)*get_global_size(0) + get_global_id(0))

__kernel void genome(__global float* sigs_in, __global float* sigs_out)
{
    __private float16 sigs;
    sigs = vload16(POS, sigs_in);

    $degregation

    $production

    vstore16(sigs, POS, sigs_out);
}
"""

Gene = namedtuple('Gene', 'reg sig_in rbs sig_out deg')

def parsegenes(genestr):
    genes = []
    for g in [genestr[i:i+5] for i in range(0, len(genestr), 5)]:
        genes.append(Gene(g[0], g[1], int(g[2]), g[3], int(g[4])))
    return genes

def degcode(genes, signal):
    rbs_deg = [(g.rbs, g.deg) for g in genes if g.sig_out == signal]
    rbs_total = float(sum(w[0] for w in rbs_deg))
    rbs_total = 1 if rbs_total == 0 else rbs_total
    deg_avg = sum((w[0]/rbs_total) * w[1] for w in rbs_deg) / 10.0
    if deg_avg == 0.0:
        return ""
    factor = 1.0 - deg_avg
    return "sigs.s{sig} *= {factor}f;".format(sig=signal, factor=factor)

def prodcode(gene):
    if gene.rbs == 0:
        return ""
    act = "activation(sigs.s{sig})".format(sig=gene.sig_in)
    if gene.reg == '-':
        act = "(1.0f - {act})".format(act=act)
    amt = "{act} * {rbs:0.1f}f".format(act=act, rbs=0.1*gene.rbs)
    return "sigs.s{out} += {amt};".format(out=gene.sig_out, amt=amt)
    
def genome_cl(genestr):
    genes = parsegenes(genestr)
    signals = [str(i) for i in range(10)] + [chr(c) for c in range(65, 71)]
    degregation = "\n    ".join([degcode(genes, s) for s in signals])
    production = "\n    ".join([prodcode(g) for g in genes])
    return Template(genome_cl_str).substitute(
        degregation=degregation, production=production)

if __name__ == "__main__":
    genestr = "+0605+0111-1505"
    print("// Genome:", genestr)
    print(genome_cl(genestr))
