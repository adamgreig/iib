from string import Template

features_cl_str = """//CL//
#define ETH ( $edge_threshold )
#define BTH ( $blob_threshold )
__constant sampler_t esampler = CLK_NORMALIZED_COORDS_FALSE |
                                CLK_ADDRESS_CLAMP_TO_EDGE   |
                                CLK_FILTER_NEAREST;
__kernel void edges(__read_only  image2d_t imgin1,
                    __read_only  image2d_t imgin2,
                    __write_only image2d_t imgout)
{
    __private ushort x = get_global_id(0);
    __private ushort y = get_global_id(1);

    __private float4 lmin = (float4)(INFINITY), lmax = (float4)(-INFINITY);
    __private float4 val;

    $minmax12

    val.s0 = lmin.s0 < -ETH && lmax.s0 > ETH ? 1.0f : 0.0f;
    val.s1 = lmin.s1 < -ETH && lmax.s1 > ETH ? 1.0f : 0.0f;
    val.s2 = lmin.s2 < -ETH && lmax.s2 > ETH ? 1.0f : 0.0f;
    val.s3 = lmin.s3 < -ETH && lmax.s3 > ETH ? 1.0f : 0.0f;

    write_imagef(imgout, (int2)(x, y), val);

}

__kernel void blobs(__read_only  image2d_t imgin1,
                    __read_only  image2d_t imgin2,
                    __read_only  image2d_t imgin3,
                    __read_only  image2d_t imgin4,
                    __write_only image2d_t imgout)
{
    __private ushort x = get_global_id(0);
    __private ushort y = get_global_id(1);

    __private float4 lmin = (float4)(INFINITY), lmax = (float4)(-INFINITY);
    __private float4 val;

    $minmax12
    $minmax23
    $minmax34

    __private float4 v = read_imagef(imgin2, esampler, (int2)(x, y)) -
                         read_imagef(imgin3, esampler, (int2)(x, y));

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
"""


def features_cl(edge_threshold=0.01, blob_threshold=0.03, width=1):
    lines12, lines23, lines34 = [], [], []
    val12a = "val = read_imagef(imgin1, esampler, (int2)(x{0:+d}, y{1:+d})) -"
    val12b = "      read_imagef(imgin2, esampler, (int2)(x{0:+d}, y{1:+d}));"
    val23a = "val = read_imagef(imgin2, esampler, (int2)(x{0:+d}, y{1:+d})) -"
    val23b = "      read_imagef(imgin3, esampler, (int2)(x{0:+d}, y{1:+d}));"
    val34a = "val = read_imagef(imgin3, esampler, (int2)(x{0:+d}, y{1:+d})) -"
    val34b = "      read_imagef(imgin4, esampler, (int2)(x{0:+d}, y{1:+d}));"
    minl = "lmin = min(lmin, val);"
    maxl = "lmax = max(lmax, val);"
    for u in reversed(range(-width, width+1)):
        for v in reversed(range(-width, width+1)):
            lines12.append(val12a.format(u, v))
            lines12.append(val12b.format(u, v))
            lines12.append(minl)
            lines12.append(maxl)
            if not (u == 0 and v == 0):
                lines23.append(val23a.format(u, v))
                lines23.append(val23b.format(u, v))
                lines23.append(minl)
                lines23.append(maxl)
            lines34.append(val34a.format(u, v))
            lines34.append(val34b.format(u, v))
            lines34.append(minl)
            lines34.append(maxl)
    lines12 = '\n    '.join(lines12)
    lines23 = '\n    '.join(lines23)
    lines34 = '\n    '.join(lines34)
    eth = "{0:0.4f}f".format(edge_threshold)
    bth = "{0:0.4f}f".format(blob_threshold)

    return Template(features_cl_str).substitute(edge_threshold=eth,
                                                blob_threshold=bth,
                                                minmax12=lines12,
                                                minmax23=lines23,
                                                minmax34=lines34)


def test():
    import os.path
    import numpy as np
    import pyopencl as cl
    import matplotlib.pyplot as plt
    from PIL import Image
    from skimage import io, data, transform
    from iib.simulation import convolution, reduction

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

    edges = np.empty((gs, gs, 4), np.float32)
    blobs = np.empty((10, gs, gs, 4), np.float32)
    count = np.empty(4, np.float32)
    space = np.empty((13, gs, gs, 4), np.float32)

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
    bufe = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))

    cl.enqueue_copy(queue, bufi, sigs, origin=(0, 0), region=(gs, gs))

    blur4.convolve_x(queue, (gs, gs), (wgs, wgs), bufi, bufb)
    blur4.convolve_y(queue, (gs, gs), (wgs, wgs), bufb, bufa)  # bufa t=2
    blur4.convolve_x(queue, (gs, gs), (wgs, wgs), bufa, bufc)
    blur4.convolve_y(queue, (gs, gs), (wgs, wgs), bufc, bufb)  # bufb t=4
    feats.edges(queue, (gs, gs), (wgs, wgs), bufa, bufb, bufc)
    cl.enqueue_copy(queue, edges, bufc, origin=(0, 0), region=(gs, gs))

    bufo = reduction.run_reduction(rdctn.reduction_sum, queue, gs, wgs,
                                   bufc, bufd, bufe)
    cl.enqueue_copy(queue, count, bufo, origin=(0, 0), region=(1, 1))
    print("Edge pixel counts:", count[:4])

    cl.enqueue_copy(queue, bufd, bufi, src_origin=(0, 0), dest_origin=(0, 0),
                    region=(gs, gs))
    buf1, buf2, buf3, buf4 = bufd, bufa, bufb, bufc
    cl.enqueue_copy(queue, space[0], buf1, origin=(0, 0), region=(gs, gs))
    cl.enqueue_copy(queue, space[1], buf2, origin=(0, 0), region=(gs, gs))
    cl.enqueue_copy(queue, space[2], buf3, origin=(0, 0), region=(gs, gs))
    for i in range(10):
        for j in range(3**i):
            bufu = buf3 if j % 2 == 0 else buf4
            bufv = buf4 if j % 2 == 0 else buf3
            blur2.convolve_x(queue, (gs, gs), (wgs, wgs), bufu, bufe)
            blur2.convolve_y(queue, (gs, gs), (wgs, wgs), bufe, bufv)
        feats.blobs(queue, (gs, gs), (wgs, wgs), buf4, buf3, buf2, buf1, bufe)
        cl.enqueue_copy(queue, blobs[i], bufe, origin=(0, 0), region=(gs, gs))
        cl.enqueue_copy(queue, space[i+3], buf4,
                        origin=(0, 0), region=(gs, gs))
        buf1, buf2, buf3, buf4 = buf2, buf3, buf4, buf1

    #for i in range(1, 12):
    if False:
        plt.subplot(4, 3, i+1)
        im1 = space[i-1, :, :, 3]
        im2 = space[i, :, :, 3]
        im = im2 - im1
        print(np.min(im), np.max(im))
        plt.imshow(im, cmap="gray")
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
        for j in range(2, 5):
            t = 3 + 3**j
            r = np.sqrt(2.0 * t)
            im = blobs[j, :, :, i]
            posns = np.transpose(im.nonzero())
            for xy in posns:
                circ = plt.Circle((xy[1], xy[0]), r, color="green", fill=False)
                ax.add_patch(circ)
    plt.show()


if __name__ == "__main__":
    print(features_cl())
    test()
