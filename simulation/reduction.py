import numpy as np
import pyopencl as cl


reduction_cl_str = """//CL//
__constant sampler_t rsampler = CLK_NORMALIZED_COORDS_FALSE |
                                CLK_ADDRESS_CLAMP_TO_EDGE   |
                                CLK_FILTER_NEAREST;
__kernel void reduction_sum(__read_only image2d_t imgin,
                            __write_only image2d_t imgout)
{
    __private ushort x = get_global_id(0);
    __private ushort y = get_global_id(1);
    __private float4 result;

    result =  read_imagef(imgin, rsampler, (int2)(x*2,   y*2));
    result += read_imagef(imgin, rsampler, (int2)(x*2+1, y*2));
    result += read_imagef(imgin, rsampler, (int2)(x*2+1, y*2+1));
    result += read_imagef(imgin, rsampler, (int2)(x*2,   y*2+1));

    write_imagef(imgout, (int2)(x, y), result);
}
"""


def reduction_sum_cl():
    return reduction_cl_str


def run_reduction(clctx, kernel, buf_in):
    """Run *kernel* on *buf_in* repeatedly and return the single result."""
    mf = cl.mem_flags
    gs, wgs = clctx.gs, clctx.wgs
    buf1 = cl.Image(clctx.ctx, mf.READ_WRITE, clctx.ifmt, (gs, gs))
    buf2 = cl.Image(clctx.ctx, mf.READ_WRITE, clctx.ifmt, (gs, gs))
    for i in range(1, int(np.log2(gs) + 1)):
        sgs = gs // (2**i)
        swg = wgs if wgs < sgs else sgs
        bufa = buf_in if i == 1 else (buf1 if i % 2 == 0 else buf2)
        bufb = buf2 if i % 2 == 0 else buf1
        kernel(clctx.queue, (sgs, sgs), (swg, swg), bufa, bufb)
    result = np.empty(4, np.float32)
    cl.enqueue_copy(clctx.queue, result, bufb, origin=(0, 0), region=(1, 1))
    buf1.release()
    buf2.release()
    return result


def test():
    from iib.simulation import CLContext
    gs, wgs = 512, 16
    sigs = np.ones((gs, gs, 4), np.float32).reshape(gs*gs*4)
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    program = cl.Program(ctx, reduction_sum_cl()).build()
    ifmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
    ibuf_in = cl.Image(ctx, mf.READ_ONLY, ifmt, (gs, gs))
    clctx = CLContext(ctx, queue, ifmt, gs, wgs)
    cl.enqueue_copy(queue, ibuf_in, sigs, origin=(0, 0), region=(gs, gs))
    print(run_reduction(clctx, program.reduction_sum, ibuf_in))

if __name__ == "__main__":
    print(reduction_sum_cl())
    test()
