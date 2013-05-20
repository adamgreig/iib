import colorsys
import numpy as np
from string import Template

colour_cl_str = """//CL//
__constant float colour_lut[768] = { $colourlut };

__kernel void colour(__read_only  image2d_t input, const uchar signal,
                     __write_only image2d_t output)
{
    __private int2 coord = (int2)(get_global_id(0), get_global_id(1));
    __private float4 colour;
    __private uint idx;
    __private union { float arr[4]; float4 vec; } data;

    data.vec = read_imagef(input, coord);
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


if __name__ == "__main__":
    print(colour_cl())
