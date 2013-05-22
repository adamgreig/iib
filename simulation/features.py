import numpy as np
import pyopencl as cl

from string import Template
from iib.simulation import convolution, reduction

features_cl_str = """//CL//
#define ETH ( $edge_threshold )
#define BTH ( $blob_threshold )
__constant sampler_t esampler = CLK_NORMALIZED_COORDS_FALSE |
                                CLK_ADDRESS_CLAMP_TO_EDGE   |
                                CLK_FILTER_NEAREST;
__kernel void subtract(__read_only  image2d_t a,
                       __read_only  image2d_t b,
                       __write_only image2d_t imgout)
{
    __private int2   p  = (int2)(get_global_id(0), get_global_id(1));
    __private float4 av = read_imagef(a, p);
    __private float4 bv = read_imagef(b, p);
    write_imagef(imgout, p, av - bv);
}

__kernel void subsample(__read_only image2d_t in, __write_only image2d_t out)
{
    __private ushort x = get_global_id(0);
    __private ushort y = get_global_id(1);
    __private float4 v = read_imagef(in, esampler, (int2)(x*2, y*2));
    write_imagef(out, (int2)(x, y), v);
}

__kernel void edges(__read_only  image2d_t l1,
                    __write_only image2d_t imgout)
{
    __private ushort x = get_global_id(0);
    __private ushort y = get_global_id(1);

    __private float4 lmin = (float4)(INFINITY), lmax = (float4)(-INFINITY);
    __private float4 val;

    $minmax1

    val.s0 = lmin.s0 < -ETH && lmax.s0 > ETH ? 1.0f : 0.0f;
    val.s1 = lmin.s1 < -ETH && lmax.s1 > ETH ? 1.0f : 0.0f;
    val.s2 = lmin.s2 < -ETH && lmax.s2 > ETH ? 1.0f : 0.0f;
    val.s3 = lmin.s3 < -ETH && lmax.s3 > ETH ? 1.0f : 0.0f;

    write_imagef(imgout, (int2)(x, y), val);

}

__kernel void blobs(__read_only  image2d_t l0,
                    __read_only  image2d_t l1,
                    __read_only  image2d_t l2,
                    __write_only image2d_t imgout)
{
    __private ushort x = get_global_id(0);
    __private ushort y = get_global_id(1);

    __private float4 lmin = (float4)(INFINITY), lmax = (float4)(-INFINITY);
    __private float4 val;

    $minmax0
    $minmax1
    $minmax2

    __private float4 v = read_imagef(l1, esampler, (int2)(x, y));

    if((v.s0 < lmin.s0 && v.s0 < -BTH) || (v.s0 > lmax.s0 && v.s0 > BTH))
        val.s0 = 1.0f; else val.s0 = 0.0f;
    if((v.s1 < lmin.s1 && v.s1 < -BTH) || (v.s1 > lmax.s1 && v.s1 > BTH))
        val.s1 = 1.0f; else val.s1 = 0.0f;
    if((v.s2 < lmin.s2 && v.s2 < -BTH) || (v.s2 > lmax.s2 && v.s2 > BTH))
        val.s2 = 1.0f; else val.s2 = 0.0f;
    if((v.s3 < lmin.s3 && v.s3 < -BTH) || (v.s3 > lmax.s3 && v.s3 > BTH))
        val.s3 = 1.0f; else val.s3 = 0.0f;

    write_imagef(imgout, (int2)(x, y), val);
}

__kernel void entropy(__read_only image2d_t in, __write_only image2d_t out)
{
    __private ushort x = get_global_id(0);
    __private ushort y = get_global_id(1);

    __private float4 val;
    __private uchar4 hist[128];
    __private ushort i;

    for(i=0; i<128; i++)
        hist[i] = (uchar4)(0);

    $countvals

    __private float4 entropy = (float4)(0.0f);
    __private float sumf = (float)($enp);
    __private float p;
    for(i=0; i<128; i++) {
        if(hist[i].s0 > 0) {
            p = (float)(hist[i].s0) / sumf;
            entropy.s0 -= p * native_log2(p);
        }
        if(hist[i].s1 > 0) {
            p = (float)(hist[i].s1) / sumf;
            entropy.s1 -= p * native_log2(p);
        }
        if(hist[i].s2 > 0) {
            p = (float)(hist[i].s2) / sumf;
            entropy.s2 -= p * native_log2(p);
        }
        if(hist[i].s3 > 0) {
            p = (float)(hist[i].s3) / sumf;
            entropy.s3 -= p * native_log2(p);
        }
    }

    write_imagef(out, (int2)(x, y), entropy);
}

__kernel void variance(__read_only image2d_t in, __write_only image2d_t out)
{
    __private int2 p = (int2)(get_global_id(0), get_global_id(1));
    __private float4 v = read_imagef(in, p);
    write_imagef(out, p, native_powr(v, (float4)(2.0f)));
}
"""


def features_cl(edge_threshold=0.01, blob_threshold=0.03, width=1, ew=3):
    lines0, lines1, lines2 = [], [], []
    val0 = "val = read_imagef(l0, esampler, (int2)((x{0:+d})*2, (y{1:+d})*2));"
    val1 = "val = read_imagef(l1, esampler, (int2)(x{0:+d}, y{1:+d}));"
    val2 = "val = read_imagef(l2, esampler, (int2)((x{0:+d})/2, (y{1:+d})/2));"
    minl = "lmin = min(lmin, val);"
    maxl = "lmax = max(lmax, val);"
    for u in reversed(range(-width, width+1)):
        for v in reversed(range(-width, width+1)):
            lines0.append(val0.format(u, v))
            lines0.append(val0.format(u, v))
            lines0.append(minl)
            lines0.append(maxl)
            if not (u == 0 and v == 0):
                lines1.append(val1.format(u, v))
                lines1.append(val1.format(u, v))
                lines1.append(minl)
                lines1.append(maxl)
            lines2.append(val2.format(u, v))
            lines2.append(val2.format(u, v))
            lines2.append(minl)
            lines2.append(maxl)
    lines0 = '\n    '.join(lines0)
    lines1 = '\n    '.join(lines1)
    lines2 = '\n    '.join(lines2)

    eth = "{0:0.4f}f".format(edge_threshold)
    bth = "{0:0.4f}f".format(blob_threshold)

    countlines = []
    c1 = "val = 128.0f * read_imagef(in, esampler, (int2)(x{0:+d}, y{1:+d}));"
    c2 = "hist[convert_uchar(val.s0)].s0++; hist[convert_uchar(val.s1)].s1++;"
    c3 = "hist[convert_uchar(val.s2)].s2++; hist[convert_uchar(val.s3)].s3++;"
    for u in reversed(range(-ew, ew+1)):
        for v in reversed(range(-ew, ew+1)):
            countlines.append(c1.format(u, v))
            countlines.append(c2)
            countlines.append(c3)
    countlines = '\n    '.join(countlines)

    return Template(features_cl_str).substitute(edge_threshold=eth,
                                                blob_threshold=bth,
                                                minmax0=lines0,
                                                minmax1=lines1,
                                                minmax2=lines2,
                                                countvals=countlines,
                                                enp=(2*ew + 1)**2)


def test():
    import matplotlib.pyplot as plt
    from skimage import io, data, transform

    gs, wgs = 256, 16

    r = transform.resize
    sigs = np.empty((gs, gs, 4), np.float32)
    sigs[:, :, 0] = r(data.coins().astype(np.float32) / 255.0, (gs, gs))
    sigs[:, :, 1] = r(data.camera().astype(np.float32) / 255.0, (gs, gs))
    sigs[:, :, 2] = r(data.text().astype(np.float32) / 255.0, (gs, gs))
    sigs[:, :, 3] = r(data.checkerboard().astype(np.float32) / 255.0, (gs, gs))
    sigs[:, :, 2] = r(io.imread("../scoring/corpus/rds/turing_001.png",
                                as_grey=True), (gs, gs))
    sigs[:, :, 3] = io.imread("../scoring/corpus/synthetic/blobs.png",
                              as_grey=True)
    sigs = sigs.reshape(gs*gs*4)

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    feats = cl.Program(ctx, features_cl()).build()
    rdctn = cl.Program(ctx, reduction.reduction_sum_cl()).build()
    blur2 = cl.Program(ctx, convolution.gaussian_cl([np.sqrt(2.0)]*4)).build()
    blur4 = cl.Program(ctx, convolution.gaussian_cl([np.sqrt(4.0)]*4)).build()

    ifmt_f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
    bufi = cl.Image(ctx, mf.READ_ONLY, ifmt_f, (gs, gs))
    bufa = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    bufb = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    bufc = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    bufd = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))

    cl.enqueue_copy(queue, bufi, sigs, origin=(0, 0), region=(gs, gs))

    entropy = np.empty(4, np.float32)
    feats.entropy(queue, (gs, gs), (wgs, wgs), bufi, bufa)
    bufo = reduction.run_reduction(rdctn.reduction_sum, queue, gs, wgs,
                                   bufa, bufb, bufc)
    cl.enqueue_copy(queue, entropy, bufo, origin=(0, 0), region=(1, 1))
    entropy /= (gs * gs)
    print("Average entropy:", entropy)

    mean = np.empty(4, np.float32)
    variance = np.empty(4, np.float32)
    bufo = reduction.run_reduction(rdctn.reduction_sum, queue, gs, wgs,
                                   bufi, bufa, bufb)
    cl.enqueue_copy(queue, mean, bufo, origin=(0, 0), region=(1, 1))
    mean /= (gs * gs)
    feats.variance(queue, (gs, gs), (wgs, wgs), bufi, bufa)
    bufo = reduction.run_reduction(rdctn.reduction_sum, queue, gs, wgs,
                                   bufa, bufb, bufc)
    cl.enqueue_copy(queue, variance, bufo, origin=(0, 0), region=(1, 1))
    variance /= (gs * gs)
    variance -= mean ** 2
    print("Variance:", variance)

    edges = np.empty((gs, gs, 4), np.float32)
    count = np.empty(4, np.float32)
    blur4.convolve_x(queue, (gs, gs), (wgs, wgs), bufi, bufb)
    blur4.convolve_y(queue, (gs, gs), (wgs, wgs), bufb, bufa)  # bufa t=2
    blur4.convolve_x(queue, (gs, gs), (wgs, wgs), bufa, bufc)
    blur4.convolve_y(queue, (gs, gs), (wgs, wgs), bufc, bufb)  # bufb t=4
    feats.subtract(queue, (gs, gs), (wgs, wgs), bufb, bufa, bufc)  # c = b - a
    feats.edges(queue, (gs, gs), (wgs, wgs), bufc, bufd)  # d = edges
    cl.enqueue_copy(queue, edges, bufd, origin=(0, 0), region=(gs, gs))

    bufo = reduction.run_reduction(rdctn.reduction_sum, queue, gs, wgs,
                                   bufd, bufa, bufb)  # a, b dirty, o = counts
    cl.enqueue_copy(queue, count, bufo, origin=(0, 0), region=(1, 1))
    print("Edge pixel counts:", count[:4])

    blobs = []
    space = []
    cl.enqueue_copy(queue, bufa, bufi, src_origin=(0, 0), dest_origin=(0, 0),
                    region=(gs, gs))
    l_prev, l_curr, g_prev = bufc, bufb, bufa
    for i in range(7):
        # Prepare next layer
        d = gs // (2**i)
        swg = wgs if wgs <= d else d
        g_blurr = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (d, d))
        g_temp = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (d, d))
        l_next = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (d, d))
        blur2.convolve_x(queue, (d, d), (swg, swg), g_prev, g_temp)
        blur2.convolve_y(queue, (d, d), (swg, swg), g_temp, g_blurr)
        feats.subtract(queue, (d, d), (swg, swg), g_blurr, g_prev, l_next)
        space.append(np.empty((d, d, 4), np.float32))
        cl.enqueue_copy(queue, space[-1], l_next, origin=(0, 0), region=(d, d))

        # Find blobs in current layer
        if i >= 2:
            d = gs // (2**(i-1))
            swg = wgs if wgs <= d else d
            out = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (d, d))
            feats.blobs(queue, (d, d), (swg, swg), l_prev, l_curr, l_next, out)
            blobs.append(np.empty((d, d, 4), np.float32))
            cl.enqueue_copy(queue, blobs[-1], out,
                            origin=(0, 0), region=(d, d))
            out.release()

        # Resize current layer to start the next layer
        d = gs // (2**(i+1))
        swg = wgs if wgs <= d else d
        g_resize = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (d, d))
        feats.subsample(queue, (d, d), (swg, swg), g_blurr, g_resize)

        # Cycle through buffers
        g_blurr.release()
        g_temp.release()
        g_prev.release()
        l_prev.release()
        g_prev = g_resize
        l_prev = l_curr
        l_curr = l_next

    #for i in range(6):
    if False:
        sspace = space[i]
        plt.subplot(2, 3, i+1)
        plt.imshow(sspace[:, :, 3], cmap="gray")
        plt.xticks([])
        plt.yticks([])

    for i in range(4):
    #if False:

        plt.subplot(4, 3, i*3+1)
        img = sigs.reshape((gs, gs, 4))[:, :, i]
        plt.imshow(img, cmap="gray")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4, 3, i*3+2)
        img = edges.reshape((gs, gs, 4))[:, :, i]
        plt.imshow(img, cmap="gray")
        plt.xticks([])
        plt.yticks([])

        ax = plt.subplot(4, 3, i*3+3)
        img = sigs.reshape((gs, gs, 4))[:, :, i]
        plt.imshow(img, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        for j in range(len(blobs)):
            sblobs = blobs[j]
            s = 2**(j+1)
            r = np.sqrt(2.0) * s
            im = sblobs[:, :, i]
            posns = np.transpose(im.nonzero()) * 2**(j+1)
            for xy in posns:
                circ = plt.Circle((xy[1], xy[0]), r, color="green", fill=False)
                ax.add_patch(circ)
    plt.show()


if __name__ == "__main__":
    print(features_cl())
    test()
