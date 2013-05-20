import scipy.stats
import numpy as np
from string import Template

diffusion_cl_str = """//CL//
#define KERNEL_N    ( $kerneln )
__constant float4    conv_kernel[KERNEL_N + 1] = { $kernel };
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE   |
                               CLK_FILTER_NEAREST;

__kernel void convolve_x(__read_only  image2d_t imgin,
                         __write_only image2d_t imgout)
{
    __private ushort x = get_global_id(0);
    __private ushort y = get_global_id(1);
    __private float4 result = (float4)(0.0f);

    $convolution_x

    write_imagef(imgout, (int2)(x, y), result);
}

__kernel void convolve_y(__read_only  image2d_t imgin,
                         __write_only image2d_t imgout)
{
    __private ushort x = get_global_id(0);
    __private ushort y = get_global_id(1);
    __private float4 result = (float4)(0.0f);

    $convolution_y

    write_imagef(imgout, (int2)(x, y), result);
}
"""


def sigmas_to_kernels(sigmas):
    if len(sigmas) != 4:
        raise ValueError("kernel_sigmas must have 4 entries")
    sigmas = np.asarray(sigmas)
    kernel_n = int(np.ceil(np.max(np.sqrt(2.0*(sigmas**2)*np.log(1000.0)))))
    kernels = []
    for s in sigmas:
        if s == 0.0:
            kernels.append([1.0] + [0.0]*kernel_n)
        else:
            kernels.append(scipy.stats.norm(scale=s).pdf(range(kernel_n+1)))
    return kernels, kernel_n


def kernels_to_cl(kernels, kernel_n):
    kernel = []
    for i in range(kernel_n + 1):
        vals = ", ".join("{0:0.8f}f".format(k[i]) for k in kernels)
        kernel.append("(float4)({0})".format(vals))
    return '\n' + ',\n'.join(kernel)


def convolution_cl(kernel_n, xy):
    out = []
    for i in range(-kernel_n, kernel_n + 1):
        istr = "{0:+d}".format(i)
        if xy == 'x':
            x, y = istr, ''
        else:
            x, y = '', istr
        out.append(
            "result += conv_kernel[{0}] * read_imagef(imgin, sampler,"
            " (int2)(x{1}, y{2}));".format(
            abs(i), x, y))
    return '\n    '.join(out)


def diffusion_cl(kernel_sigmas):
    kernels, kernel_n = sigmas_to_kernels(kernel_sigmas)
    kernel = kernels_to_cl(kernels, kernel_n)
    convolution_x = convolution_cl(kernel_n, 'x')
    convolution_y = convolution_cl(kernel_n, 'y')
    return Template(diffusion_cl_str).substitute(kerneln=kernel_n,
                                                 kernel=kernel,
                                                 convolution_x=convolution_x,
                                                 convolution_y=convolution_y)

if __name__ == "__main__":
    sigmas = [1.0, 2.0, 3.0, 5.0]
    print("// Sigmas:", sigmas)
    print(diffusion_cl(sigmas, 128))
