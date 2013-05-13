import scipy.stats
import numpy as np
from string import Template

diffusion_cl_str = """//CL//
#define KERNEL_N   ( $kerneln )
#define WG_SIZE    ( $wgsize  )
#define GCOL (get_global_id(0))
#define GROW (get_global_id(1))
#define GWIDTH (get_global_size(0))
#define GHEIGHT (get_global_size(1))
#define WGWIDTH (get_local_size(0))
#define WGHEIGHT (get_local_size(1))
#define GPOS (GROW * GWIDTH + GCOL)
#define LIDX (get_local_id(longdim))
#define WUIDX (get_global_id(longdim))
__constant float16 conv_kernel[KERNEL_N + 1] = { $kernel };

__kernel void diffusion(__global float* sigs_in, __global float* sigs_out)
{
    __private uchar longdim;
    __local ushort wgidx, wgsize, stepsize, gsize;

    // Set up variables to allow for tall/wide diffusion
    if(WGWIDTH == 1) {
        // Tall kernel
        longdim = 1;
        wgidx = GCOL;           // Same for whole WG
        gsize = GHEIGHT;        // Relevant grid size
        wgsize = WGHEIGHT;      // WG's larger dimension
        stepsize = gsize;       // How spaced consecutive items are in sigs_in
    } else {
        // Wide kernel
        longdim = 0;
        wgidx = GROW;           // Same for whole WG
        gsize = GWIDTH;         // Relevant grid size
        wgsize = WGWIDTH;       // WG's larger dimension
        stepsize = 1;           // How spaced consecutive items are in sigs_in
    }

    // Space to store the entire workgroup's results during computation
    __local float16 result[WG_SIZE];
    result[LIDX] = (float16)(0.0f);

    // Space to store the entire workgroup's input plus an apron to either side
    __local float16 wg_sigs[WG_SIZE + 2*KERNEL_N];

    // Copy this workitem's memory
    wg_sigs[LIDX + KERNEL_N] = vload16(GPOS, sigs_in);

    // Edge workitems also copy apron
    if(LIDX < KERNEL_N) {
        if(WUIDX >= KERNEL_N) {
            // When not at the grid edge, copy the apron cells
            wg_sigs[LIDX] = vload16(GPOS - KERNEL_N * stepsize, sigs_in);
        } else {
            // When at the grid edge, extend the edge values into the apron
            wg_sigs[LIDX] = vload16(GPOS - LIDX*stepsize, sigs_in);
        }
    } else if(LIDX >= wgsize - KERNEL_N) {
        if(WUIDX < gsize - KERNEL_N) {
            // When not at the grid edge, copy the apron cells
            wg_sigs[LIDX + 2*KERNEL_N] =
                vload16(GPOS + KERNEL_N*stepsize, sigs_in);
        } else {
            // When at the grid edge, extend the edge values into the apron
            wg_sigs[LIDX + 2*KERNEL_N] = 0.0f;
            wg_sigs[LIDX + 2*KERNEL_N] =
                vload16(GPOS + (wgsize - LIDX - 1)*stepsize, sigs_in);
        }
    }

    // Wait for each work group to catch up
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform convolution
    $convolution

    // Store result
    vstore16(result[LIDX], GPOS, sigs_out);
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
        out.append(
            "result[LIDX] += conv_kernel[{0}] * wg_sigs[LIDX + {1}];".format(
            abs(i - kernel_n), i))
    return '\n    '.join(out)

def diffusion_cl(kernel_sigmas, wg_size):
    kernels, kernel_n = sigmas_to_kernels(kernel_sigmas)
    kernel = kernels_to_cl(kernels, kernel_n)
    convolution = convolution_cl(kernel_n)
    return Template(diffusion_cl_str).substitute(kerneln=kernel_n,
        wgsize=wg_size, kernel=kernel, convolution=convolution)

if __name__ == "__main__":
    sigmas = [0.8, 1.0, 1.2, 1.5, 2.0, 2.0, 5.0, 5.0] * 2
    print("// Sigmas:", sigmas)
    print(diffusion_cl(sigmas))
