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
    __private float4 av = read_imagef(a, esampler, p);
    __private float4 bv = read_imagef(b, esampler, p);
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
    __private ushort w0 = get_local_size(0);
    __private ushort w1 = get_local_size(1);
    __local uint hist_s0[256];
    __local uint hist_s1[256];
    __local uint hist_s2[256];
    __local uint hist_s3[256];

    __private float wgss = convert_float(w0 * w1);
    __private ushort idx = get_local_id(0) * w0 + get_local_id(1);
    __private float4 val = 256.0f * read_imagef(in, esampler, (int2)(x, y));

    hist_s0[idx] = 0;
    hist_s1[idx] = 0;
    hist_s2[idx] = 0;
    hist_s3[idx] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_inc(&(hist_s0[convert_uchar(val.s0)]));
    atomic_inc(&(hist_s1[convert_uchar(val.s1)]));
    atomic_inc(&(hist_s2[convert_uchar(val.s2)]));
    atomic_inc(&(hist_s3[convert_uchar(val.s3)]));

    barrier(CLK_LOCAL_MEM_FENCE);

    __private float4 entropy = (float4)(0.0f);
    __private float p;

    // avoid branching for the p=0 case by making p very very small
    p = ((float)(hist_s0[idx]) / wgss) + 0.000001f;
    entropy.s0 = -p * native_log2(p);
    p = ((float)(hist_s1[idx]) / wgss) + 0.000001f;
    entropy.s1 = -p * native_log2(p);
    p = ((float)(hist_s2[idx]) / wgss) + 0.000001f;
    entropy.s2 = -p * native_log2(p);
    p = ((float)(hist_s3[idx]) / wgss) + 0.000001f;
    entropy.s3 = -p * native_log2(p);

    write_imagef(out, (int2)(x, y), entropy);
}

__kernel void variance(__read_only image2d_t in, __write_only image2d_t out)
{
    __private int2 p = (int2)(get_global_id(0), get_global_id(1));
    __private float4 v = read_imagef(in, esampler, p);
    write_imagef(out, p, native_powr(v, (float4)(2.0f)));
}
"""


def features_cl(edge_threshold=0.01, blob_threshold=0.03, width=1, ew=3):
    lines0, lines1, lines2 = [], [], []
    val0 = "val = read_imagef(l0, esampler, (int2)((x{0:+d})*2, (y{1:+d})*2));"
    val1 = "val = read_imagef(l1, esampler, (int2)(x{0:+d}, y{1:+d}));"
    val2 = "val = read_imagef(l2, esampler, (int2)((x{0:+d})/2, (y{1:+d})/2));"
    minmaxl = "lmin = min(lmin, val); lmax = max(lmax, val);"
    for u in reversed(range(-width, width+1)):
        for v in reversed(range(-width, width+1)):
            lines0.append(val0.format(u, v))
            lines0.append(val0.format(u, v))
            lines0.append(minmaxl)
            if not (u == 0 and v == 0):
                lines1.append(val1.format(u, v))
                lines1.append(val1.format(u, v))
                lines1.append(minmaxl)
            lines2.append(val2.format(u, v))
            lines2.append(val2.format(u, v))
            lines2.append(minmaxl)
    lines0 = '\n    '.join(lines0)
    lines1 = '\n    '.join(lines1)
    lines2 = '\n    '.join(lines2)

    eth = "{0:0.4f}f".format(edge_threshold)
    bth = "{0:0.4f}f".format(blob_threshold)

    return Template(features_cl_str).substitute(edge_threshold=eth,
                                                blob_threshold=bth,
                                                minmax0=lines0,
                                                minmax1=lines1,
                                                minmax2=lines2)


def get_variance(clctx, features, reductions, buf_in):
    """Using the *features* and *reductions* programs, find Var[*buf_in*]."""
    gs, wgs = clctx.gs, clctx.wgs
    buf = cl.Image(clctx.ctx, cl.mem_flags.READ_WRITE, clctx.ifmt, (gs, gs))
    mean = reduction.run_reduction(clctx, reductions.reduction_sum, buf_in)
    mean /= gs * gs
    features.variance(clctx.queue, (gs, gs), (wgs, wgs), buf_in, buf)
    variance = reduction.run_reduction(clctx, reductions.reduction_sum, buf)
    variance /= gs * gs
    variance -= mean ** 2
    buf.release()
    return variance


def get_entropy(clctx, features, reductions, buf_in):
    """Using the *features* and *reductions* programs, find H[*buf_in*]."""
    gs, wgs = clctx.gs, clctx.wgs
    buf = cl.Image(clctx.ctx, cl.mem_flags.READ_WRITE, clctx.ifmt, (gs, gs))
    features.entropy(clctx.queue, (gs, gs), (wgs, wgs), buf_in, buf)
    entropy = reduction.run_reduction(clctx, reductions.reduction_sum, buf)
    entropy /= ((gs * gs) / (wgs * wgs))
    buf.release()
    return entropy


def get_edges(clctx, features, reductions, blurs, buf_in, summarise=True):
    """
    Using the *features* and *reductions* programs, and *blurs* program with
    sigma=2.0, find all edge pixels in *buf_in* and return the count.
    """
    gs, wgs = clctx.gs, clctx.wgs
    bufa = cl.Image(clctx.ctx, cl.mem_flags.READ_WRITE, clctx.ifmt, (gs, gs))
    bufb = cl.Image(clctx.ctx, cl.mem_flags.READ_WRITE, clctx.ifmt, (gs, gs))
    bufc = cl.Image(clctx.ctx, cl.mem_flags.READ_WRITE, clctx.ifmt, (gs, gs))

    blurs.convolve_x(clctx.queue, (gs, gs), (wgs, wgs), buf_in, bufb)
    blurs.convolve_y(clctx.queue, (gs, gs), (wgs, wgs), bufb, bufa)
    blurs.convolve_x(clctx.queue, (gs, gs), (wgs, wgs), bufa, bufc)
    blurs.convolve_y(clctx.queue, (gs, gs), (wgs, wgs), bufc, bufb)

    features.subtract(clctx.queue, (gs, gs), (wgs, wgs), bufb, bufa, bufc)
    features.edges(clctx.queue, (gs, gs), (wgs, wgs), bufc, bufa)
    counts = reduction.run_reduction(clctx, reductions.reduction_sum, bufa)

    if not summarise:
        edges = np.empty((gs, gs, 4), np.float32)
        cl.enqueue_copy(clctx.queue, edges, bufa,
                        origin=(0, 0), region=(gs, gs))

    bufa.release()
    bufb.release()
    bufc.release()

    if summarise:
        return counts
    else:
        return edges


def get_blobs(clctx, features, reductions, blurs, buf_in, summarise=True):
    """
    Using the *features* and *reductions* programs, and *blurs* program with
    sigma=sqrt(2.0), find all the blobs in *buf_in* at five scales and return
    the count at each scale.
    """
    counts = np.empty((5, 4), np.float32)
    gs, wgs = clctx.gs, clctx.wgs
    mf = cl.mem_flags
    bufa = cl.Image(clctx.ctx, cl.mem_flags.READ_WRITE, clctx.ifmt, (gs, gs))
    cl.enqueue_copy(clctx.queue, bufa, buf_in, src_origin=(0, 0),
                    dest_origin=(0, 0), region=(gs, gs))
    l_prev, l_curr, g_prev = None, None, bufa

    if not summarise:
        blobs = []

    for i in range(7):
        # Prepare next layer
        d = gs // (2**i)
        swg = wgs if wgs <= d else d
        g_blurr = cl.Image(clctx.ctx, mf.READ_WRITE, clctx.ifmt, (d, d))
        g_temp = cl.Image(clctx.ctx, mf.READ_WRITE, clctx.ifmt, (d, d))
        l_next = cl.Image(clctx.ctx, mf.READ_WRITE, clctx.ifmt, (d, d))
        blurs.convolve_x(clctx.queue, (d, d), (swg, swg), g_prev, g_temp)
        blurs.convolve_y(clctx.queue, (d, d), (swg, swg), g_temp, g_blurr)
        features.subtract(clctx.queue, (d, d), (swg, swg),
                          g_blurr, g_prev, l_next)

        # Find blobs in current layer
        if i >= 2:
            d = gs // (2**(i-1))
            swg = wgs if wgs <= d else d
            out = cl.Image(clctx.ctx, mf.READ_WRITE, clctx.ifmt, (d, d))
            features.blobs(clctx.queue, (d, d), (swg, swg),
                           l_prev, l_curr, l_next, out)
            rs = reductions.reduction_sum
            counts[i-2] = reduction.run_reduction(clctx, rs, out)
            if not summarise:
                blobs.append(np.empty((d, d, 4), np.float32))
                cl.enqueue_copy(clctx.queue, blobs[-1], out,
                                origin=(0, 0), region=(d, d))
            out.release()

        # Resize current layer to start the next layer
        d = gs // (2**(i+1))
        swg = wgs if wgs <= d else d
        g_resize = cl.Image(clctx.ctx, mf.READ_WRITE, clctx.ifmt, (d, d))
        features.subsample(clctx.queue, (d, d), (swg, swg), g_blurr, g_resize)

        # Cycle through buffers
        g_blurr.release()
        g_temp.release()
        g_prev.release()
        if l_prev:
            l_prev.release()
        g_prev = g_resize
        l_prev = l_curr
        l_curr = l_next

    if summarise:
        return counts
    else:
        return blobs


def get_features(clctx, features, reductions, blur2, blur4, buf_in):
    """
    Return the 8-dimensional feature vector of *buf_in*.
    *features* and *reductions* are the eponymous programs.
    *blur2* and *blur4* are convolution with sigma=sqrt(2), 2 respectively.
    """
    edge_counts = get_edges(clctx, features, reductions, blur4, buf_in)
    blob_counts = get_blobs(clctx, features, reductions, blur2, buf_in)
    variance = get_variance(clctx, features, reductions, buf_in)
    entropy = get_entropy(clctx, features, reductions, buf_in)
    return np.vstack((edge_counts, blob_counts, variance, entropy))


def test():
    import matplotlib.pyplot as plt
    from skimage import io, data, transform
    from iib.simulation import CLContext

    gs, wgs = 256, 16

    # Load some test data
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
    #sq = np.arange(256).astype(np.float32).reshape((16, 16)) / 255.0
    #sigs[:, :, 0] = np.tile(sq, (16, 16))
    sigs = sigs.reshape(gs*gs*4)

    # Set up OpenCL
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    ifmt_f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
    bufi = cl.Image(ctx, mf.READ_ONLY, ifmt_f, (gs, gs))
    cl.enqueue_copy(queue, bufi, sigs, origin=(0, 0), region=(gs, gs))
    clctx = CLContext(ctx, queue, ifmt_f, gs, wgs)

    # Compile the kernels
    feats = cl.Program(ctx, features_cl()).build()
    rdctn = cl.Program(ctx, reduction.reduction_sum_cl()).build()
    blur2 = cl.Program(ctx, convolution.gaussian_cl([np.sqrt(2.0)]*4)).build()
    blur4 = cl.Program(ctx, convolution.gaussian_cl([np.sqrt(4.0)]*4)).build()

    entropy = get_entropy(clctx, feats, rdctn, bufi)
    print("Average entropy:", entropy)

    variance = get_variance(clctx, feats, rdctn, bufi)
    print("Variance:", variance)

    edges = get_edges(clctx, feats, rdctn, blur4, bufi, summarise=False)
    edge_counts = get_edges(clctx, feats, rdctn, blur4, bufi)
    print("Edge pixel counts:", edge_counts)

    blobs = get_blobs(clctx, feats, rdctn, blur2, bufi, summarise=False)

    features = get_features(clctx, feats, rdctn, blur2, blur4, bufi)
    print("Feature vector:")
    print(features)

    # Plot the edges and blobs
    for i in range(4):
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


def profile():
    import time
    from iib.simulation import CLContext
    from skimage import io, data, transform
    gs, wgs = 256, 16

    # Load some test data
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

    # Set up OpenCL
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    ifmt_f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
    bufi = cl.Image(ctx, mf.READ_ONLY, ifmt_f, (gs, gs))
    cl.enqueue_copy(queue, bufi, sigs, origin=(0, 0), region=(gs, gs))
    clctx = CLContext(ctx, queue, ifmt_f, gs, wgs)

    # Compile the kernels
    feats = cl.Program(ctx, features_cl()).build()
    rdctn = cl.Program(ctx, reduction.reduction_sum_cl()).build()
    blur2 = cl.Program(ctx, convolution.gaussian_cl([np.sqrt(2.0)]*4)).build()
    blur4 = cl.Program(ctx, convolution.gaussian_cl([np.sqrt(4.0)]*4)).build()

    iters = 500
    t0 = time.time()
    for i in range(iters):
        get_features(clctx, feats, rdctn, blur2, blur4, bufi)
    print((time.time() - t0)/iters)

if __name__ == "__main__":
    print(features_cl())
    test()
    #profile()
