from string import Template

edges_cl_str = """//CL//
#define TH ( $threshold )
__constant sampler_t esampler = CLK_NORMALIZED_COORDS_FALSE |
                                CLK_ADDRESS_CLAMP_TO_EDGE   |
                                CLK_FILTER_NEAREST;
__kernel void edges(__read_only  image2d_t imgin1,
                    __read_only  image2d_t imgin2,
                    __write_only image2d_t imgout)
{
    __private ushort x = get_global_id(0);
    __private ushort y = get_global_id(1);

    __private float4 lmin = INFINITY, lmax = -INFINITY;
    __private float4 val;
    
    $findminmax

    val.s0 = lmin.s0 < -TH && lmax.s0 > TH ? 1.0f : 0.0f;
    val.s1 = lmin.s1 < -TH && lmax.s1 > TH ? 1.0f : 0.0f;
    val.s2 = lmin.s2 < -TH && lmax.s2 > TH ? 1.0f : 0.0f;
    val.s3 = lmin.s3 < -TH && lmax.s3 > TH ? 1.0f : 0.0f;

    write_imagef(imgout, (int2)(x, y), val);

}
"""

def edges_cl(threshold=0.01, width=1):
    windowlines = []
    vall = "val = read_imagef(imgin1, esampler, (int2)(x{0:+d}, y{1:+d})) -\n"\
           "          read_imagef(imgin2, esampler, (int2)(x{0:+d}, y{1:+d}));"
    minl = "lmin = min(lmin, val);"
    maxl = "lmax = max(lmax, val);"
    for u in reversed(range(-width, width+1)):
        for v in reversed(range(-width, width+1)):
            if not (u == 0 and v == 0):
                windowlines.append(vall.format(u, v))
                windowlines.append(minl)
                windowlines.append(maxl)
    windowlines = '\n    '.join(windowlines)
    th = "{0:0.4f}f".format(threshold)

    return Template(edges_cl_str).substitute(threshold=th,
                                             findminmax=windowlines)


def test():
    import time
    import os.path
    import numpy as np
    import pyopencl as cl
    import matplotlib.pyplot as plt
    from PIL import Image
    from skimage import data, transform
    from iib.simulation import diffusion
    gs, wgs = 256, 16
    r = transform.resize
    coins = r(data.coins().astype(np.float32) / 255.0, (gs, gs))
    camera = r(data.camera().astype(np.float32) / 255.0, (gs, gs))
    text = r(data.text().astype(np.float32) / 255.0, (gs, gs))
    cboard = r(data.checkerboard().astype(np.float32) / 255.0, (gs, gs))
    sigs = np.empty((gs, gs, 4), np.float32)
    sigs[:, :, 0] = coins
    sigs[:, :, 1] = camera
    sigs[:, :, 2] = text
    sigs[:, :, 3] = cboard
    sigs = sigs.reshape(gs*gs*4)
    fpath = os.path.dirname(os.path.abspath(__file__))
    for i in range(4):
        subimg = sigs.reshape((gs, gs, 4))[:, :, i]
        subimg = (subimg * 255.0).astype(np.uint8)
        img = Image.fromarray(subimg)
        path = fpath + "/output/edges_before_{0}.png".format(i)
        img.save(path)

    edges = np.empty((gs, gs, 4), np.float32)
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    program = cl.Program(ctx, edges_cl()).build()
    ifmt_f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
    ibuf_in = cl.Image(ctx, mf.READ_ONLY, ifmt_f, (gs, gs))
    ibuf_a = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    ibuf_b = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    ibuf_c = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    ibuf_out = cl.Image(ctx, mf.WRITE_ONLY, ifmt_f, (gs, gs))
    cl.enqueue_copy(queue, ibuf_in, sigs, origin=(0, 0), region=(gs, gs))

    print(time.time())
    for i in range(1000):
        blur1 = cl.Program(ctx, diffusion.diffusion_cl([3.0]*4)).build()
        blur2 = cl.Program(ctx, diffusion.diffusion_cl([1.9]*4)).build()
        blur1.convolve_x(queue, (gs, gs), (wgs, wgs), ibuf_in, ibuf_b)
        blur1.convolve_y(queue, (gs, gs), (wgs, wgs), ibuf_b, ibuf_a)
        blur2.convolve_x(queue, (gs, gs), (wgs, wgs), ibuf_in, ibuf_c)
        blur2.convolve_y(queue, (gs, gs), (wgs, wgs), ibuf_c, ibuf_b)
        program.edges(queue, (gs, gs), (wgs, wgs), ibuf_a, ibuf_b, ibuf_out)
        cl.enqueue_copy(queue, edges, ibuf_out, origin=(0, 0), region=(gs, gs))
    print(time.time())

    for i in range(4):
        subimg = edges.reshape((gs, gs, 4))[:, :, i]
        subimg = (subimg * 255.0).astype(np.uint8)
        img = Image.fromarray(subimg)
        path = fpath + "/output/edges_after_{0}.png".format(i)
        img.save(path)

        plt.subplot(4, 2, i*2+1)
        img = sigs.reshape((gs, gs, 4))[:, :, i]
        plt.imshow(img, cmap="gray")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4, 2, i*2+2)
        img = edges.reshape((gs, gs, 4))[:, :, i]
        plt.imshow(img, cmap="gray")
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == "__main__":
    print(edges_cl())
    test()
