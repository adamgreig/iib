import colorsys
import numpy as np
import pyopencl as cl
from PIL import Image
from string import Template

colour_cl_str = """//CL//
__constant float colour_lut[768] = { $colourlut };
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE   |
                               CLK_FILTER_NEAREST;

__kernel void colour(__read_only  image2d_t input, const uchar signal,
                     __write_only image2d_t output)
{
    __private int2 coord = (int2)(get_global_id(0), get_global_id(1));
    __private float4 colour;
    __private uint idx;
    __private union { float arr[4]; float4 vec; } data;

    data.vec = read_imagef(input, sampler, coord);
    idx = (uint)(clamp(data.arr[signal], 0.0f, 1.0f) * 255.0f);
    colour.x = colour_lut[idx * 3    ];
    colour.y = colour_lut[idx * 3 + 1];
    colour.z = colour_lut[idx * 3 + 2];
    colour.w = 1.0f;
    write_imagef(output, coord, colour);
}
"""


def colour_cl():
    h2r = colorsys.hsv_to_rgb
    colour_lut = [h2r((1.0-(x/255.0)) * 2.0/3.0, 1.0, 1.0) for x in range(256)]
    colour_lut = np.array(colour_lut, np.float32).reshape(256*3).astype(float)
    colour_lut = ", ".join("{0:0.6f}f".format(k) for k in colour_lut)
    return Template(colour_cl_str).substitute(colourlut=colour_lut)


def dump_colour_image(clctx, colours, bufi, channel, fpath):
    gs, wgs = clctx.gs, clctx.wgs
    ifmt_u = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)
    buf = cl.Image(clctx.ctx, cl.mem_flags.WRITE_ONLY, ifmt_u, (gs, gs))
    img = np.empty(gs*gs*4, np.uint8)
    channel = str(chr(channel)).encode()
    colours.colour(clctx.queue, (gs, gs), (wgs, wgs), bufi, channel, buf)
    cl.enqueue_copy(clctx.queue, img, buf, origin=(0, 0), region=(gs, gs))
    img = img.reshape((gs, gs, 4))
    img = Image.fromarray(img)
    img.save(fpath)
    buf.release()


def test():
    from iib.simulation import CLContext
    gs, wgs = 256, 16
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    ifmt_f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
    bufi = cl.Image(ctx, cl.mem_flags.READ_ONLY, ifmt_f, (gs, gs))
    colours = cl.Program(ctx, colour_cl()).build()
    clctx = CLContext(ctx, queue, ifmt_f, gs, wgs)

    x = np.arange(-128., 128.)/30
    xx = np.outer(x, x)
    y = np.sinc(xx)
    sigs = np.empty((gs, gs, 4), np.float32)
    sigs[:, :, 0] = y
    sigs = sigs.reshape(gs*gs*4)
    cl.enqueue_copy(queue, bufi, sigs, origin=(0, 0), region=(gs, gs))

    dump_colour_image(clctx, colours, bufi, 0, "testimage.png")
    print("Test image dumped to ./testimage.png")

if __name__ == "__main__":
    print(colour_cl())
    test()
