from collections import namedtuple
from string import Template

genome_cl_str = """//CL//
// A slightly leaky activation function that saturates at about x=1.0
#define activation(x) native_recip(1.0f + native_exp(5.0f - 10.0f * x))
__constant sampler_t gsampler = CLK_NORMALIZED_COORDS_FALSE |
                                CLK_ADDRESS_CLAMP_TO_EDGE   |
                                CLK_FILTER_NEAREST;

__kernel void genome(__read_only  image2d_t sigs_in_a,
                     __read_only  image2d_t sigs_in_b,
                     __write_only image2d_t sigs_out_a,
                     __write_only image2d_t sigs_out_b)
{
    __private int2 pos = (int2)(get_global_id(0), get_global_id(1));
    __private float8 sigs;
    sigs.lo = read_imagef(sigs_in_a, gsampler, pos);
    sigs.hi = read_imagef(sigs_in_b, gsampler, pos);

    $degregation

    $production

    write_imagef(sigs_out_a, pos, sigs.lo);
    write_imagef(sigs_out_b, pos, sigs.hi);
}
"""

Gene = namedtuple('Gene', 'reg sig_in rbs sig_out deg')


def get_used_genes(genestr):
    genes = parsegenes(genestr)
    used = {int(s, 16) for sl in ((g[3],) for g in genes) for s in sl}
    return list(used)


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
    signals = [str(i) for i in range(8)]
    degregation = "\n    ".join([degcode(genes, s) for s in signals])
    production = "\n    ".join([prodcode(g) for g in genes])
    return Template(genome_cl_str).substitute(
        degregation=degregation, production=production)

if __name__ == "__main__":
    genestr = "+0605+0111-1505"
    print("// Genome:", genestr)
    print(genome_cl(genestr))
