reduction_cl_str = """//CL//
__kernel void reduction_sum(__read_only image2d_t imgin,
                            __write_only image2d_t imgout)
{
    __private ushort x = get_global_id(0);
    __private ushort y = get_global_id(1);
    __private float4 result;

    result =  read_imagef(imgin, (int2)(x*2,   y*2));
    result += read_imagef(imgin, (int2)(x*2+1, y*2));
    result += read_imagef(imgin, (int2)(x*2+1, y*2+1));
    result += read_imagef(imgin, (int2)(x*2,   y*2+1));

    write_imagef(imgout, (int2)(x, y), result);
}
"""


def reduction_add_cl():
    return reduction_cl_str


def test():
    import numpy as np
    import pyopencl as cl
    gs, wgs = 2048, 16
    sigs = np.ones((gs, gs, 4), np.float32).reshape(gs*gs*4)
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    program = cl.Program(ctx, reduction_add_cl()).build()
    ifmt_f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
    ibuf_in = cl.Image(ctx, mf.READ_ONLY, ifmt_f, (gs, gs))
    ibuf_a = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    ibuf_b = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    cl.enqueue_copy(queue, ibuf_in, sigs, origin=(0, 0), region=(gs, gs))
    for i in range(1, int(np.log2(gs) + 1)):
        sgs = gs // (2**i)
        swg = wgs if wgs <= sgs else sgs
        ibuf_1 = ibuf_in if i == 1 else (ibuf_a if i % 2 == 0 else ibuf_b)
        ibuf_2 = ibuf_b if i % 2 == 0 else ibuf_a
        program.reduction_sum(queue, (sgs, sgs), (swg, swg), ibuf_1, ibuf_2)
    cl.enqueue_copy(queue, sigs, ibuf_a, origin=(0, 0), region=(1, 1))
    print(sigs[:4])

if __name__ == "__main__":
    print(reduction_add_cl())
    test()
