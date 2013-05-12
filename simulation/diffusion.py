import scipy.stats
import numpy as np
from string import Template

diffusion_cl_str = """//CL//
#define KERNEL_N   ( $kerneln )
__constant float16 conv_kernel[KERNEL_N + 1] = { $kernel };

__kernel void diffuse(__global float* sigs_in, __global float* sigs_out)
{
    __private ushort lidx, wuidx;
    __local ushort wgidx, wgsize, gsize, stepsize;
    __private ushort gcol = get_global_id(0);
    __private ushort grow = get_global_id(1);
    __local ushort wgwidth = get_local_size(0);
    __local ushort wgheight = get_local_size(1);
    __local ushort gwidth = get_global_size(0);
    __local ushort gheight = get_global_size(1);
    __private ushort gpos = grow * gwidth + gcol;
    __private float16 result = (float16)(0.0f);

    // Set up variables to allow for tall/wide diffusion
    if(wgwidth == 1) {
        // Tall kernel
        lidx = get_local_id(1); // Local index
        wuidx = grow;           // Differs in each WU
        wgidx = gcol;           // Same for whole WG
        gsize = gheight;        // Relevant grid size
        wgsize = wgheight;      // WG's larger dimension
        stepsize = gsize;       // How spaced consecutive items are in sigs_in
    } else {
        // Wide kernel
        lidx = get_local_id(0); // Local index
        wuidx = gcol;           // Differs in each WU
        wgidx = grow;           // Same for whole WG
        gsize = gwidth;         // Relevant grid size
        wgsize = wgwidth;       // WG's larger dimension
        stepsize = 1;           // How spaced consecutive items are in sigs_in
    }

    // Space to store the entire workgroup's input plus an apron to either side
    __local float16 wg_sigs[wgsize + 2*KERNEL_N];

    // Copy this workitem's memory
    wg_sigs[lidx + KERNEL_N] = vload16(gpos, sigs_in);

    // Edge workitems also copy apron
    if(lidx < KERNEL_N) {
        if(wuidx >= KERNEL_N) {
            // When not at the grid edge, copy the apron cells
            wg_sigs[lidx] = vload16(gpos - KERNEL_N*stepsize, sigs_in);
        } else {
            // When at the grid edge, extend the edge values into the apron
            wg_sigs[lidx] = vload16(gpos - lidx*stepsize, sigs_in);
        }
    } else if(lidx >= wgsize - KERNEL_N) {
        if(wuidx < gsize - KERNEL_N) {
            // When not at the grid edge, copy the apron cells
            wg_sigs[lidx + 2*KERNEL_N] = vload16(gpos + KERNEL_N*stepsize, sigs_in);
        } else {
            // When at the grid edge, extend the edge values into the apron
            wg_sigs[lidx + 2*KERNEL_N] = vload16(gpos + (KERNEL_N - lidx)*stepsize, sigs_in);
        }
    }

    // Wait for each work group to catch up
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform convolution
    $convolution

    // Store result
    vstore16(result, gpos, sigs_out);
}
"""

def sigmas_to_kernels(sigmas):
    if len(sigmas) != 16:
        raise ValueError("kernel_sigmas must have 16 entries")
    sigmas = np.asarray(sigmas)
    kernel_n = int(np.ceil(np.max(np.sqrt(2.0*(sigmas**2)*np.log(1000.0)))))
    kernels = []
    for sigma in sigmas:
        kernels.append(scipy.stats.norm(scale=sigma).pdf(range(kernel_n+1)))
    return kernels, kernel_n

def kernels_to_cl(kernels, kernel_n):
    kernel = []
    for i in range(kernel_n + 1):
        vals = ", ".join("{0:0.8f}f".format(k[i]) for k in kernels)
        kernel.append("(float16)({0})".format(vals))
    return '\n' + ',\n'.join(kernel)

def convolution_cl(kernel_n):
    out = []
    for i in range(2*kernel_n + 1):
        out.append("result += conv_kernel[{0}] * wg_sigs[lidx + {1}];".format(
                   abs(i - kernel_n), i))
    return '\n    '.join(out)

def diffusion_cl(kernel_sigmas):
    kernels, kernel_n = sigmas_to_kernels(kernel_sigmas)
    kernel = kernels_to_cl(kernels, kernel_n)
    convolution = convolution_cl(kernel_n)
    return Template(diffusion_cl_str).substitute(
        kerneln=kernel_n, kernel=kernel, convolution=convolution)

if __name__ == "__main__":
    sigmas = [0.8, 1.0, 1.2, 1.5, 2.0, 2.0, 5.0, 5.0] * 2
    print("// Sigmas:", sigmas)
    print(diffusion_cl(sigmas))
