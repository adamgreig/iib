__kernel void cell(__global float* light, __global float* sigs_in,
                   __global float* sigs_out, __global float* out)
{
    uint col = get_global_id(0);
    uint row = get_global_id(1);

    float cell_light = light[  row*512 + col];
    float cell_sig   = sigs_in[row*512 + col];

    out[row*512 + col] = cell_light * cell_sig;
    sigs_out[row*512 + col] = (cell_sig * 0.8f) + (1.0f - cell_light);
}

__kernel void diffuse(__global float* sigs_in, __global float* sigs_out)
{
    uint idx, wuidx, wgidx, width, height;
    uint col = get_global_id(0);
    uint row = get_global_id(1);

    // Either this kernel is called with wg size (1, 256) or (256, 1)
    if(get_local_size(0) == 1) {
        // Tall kernel
        idx = get_local_id(1); // the incrementing local index
        wuidx = row; // different for each work unit
        wgidx = col; // same for whole workgroup
        width = 1;
        height = 512;
    } else {
        // Wide kernel
        idx = get_local_id(0); // the incrementing local index
        wuidx = col; // different for each work unit
        wgidx = row; // same for whole workgroup
        width = 512;
        height = 1;
    }

    // Store the entire workgroup, plus 3 cells of apron on either side
    __local float wg_sigs[256 + 3 + 3];

    // Copy over the corresponding part of the main memory
    wg_sigs[idx + 3] = sigs_in[row*512 + col];

    // For work units at the edge of the group, also copy over the apron
    if(wuidx <= 2) {
        // Set the beyond-grid cells to the grid-edge value
        wg_sigs[wuidx] = sigs_in[wgidx * width];
    } else if(wuidx >= 253 && wuidx <= 255) {
        // Set the overlapping cells to the proper grid values
        wg_sigs[wuidx + 6] = sigs_in[(wuidx + 3) * height + wgidx * width];
    } else if(wuidx >= 256 && wuidx <= 258) {
        // Set the overlapping cells to the proper grid values
        wg_sigs[wuidx - 256] = sigs_in[(wuidx - 3) * height + wgidx * width];
    } else if(wuidx >= 509) {
        // Set the beyond-grid cells to the grid-edge value
        wg_sigs[wuidx - 250] = sigs_in[height * 511 + wgidx * width];
    }

    // Wait for each work group to catch up
    barrier(CLK_LOCAL_MEM_FENCE);

    // Convolve the kernel
    sigs_out[row*512 + col] = (
        conv_kernel[3] * wg_sigs[idx    ] +
        conv_kernel[2] * wg_sigs[idx + 1] +
        conv_kernel[1] * wg_sigs[idx + 2] +
        conv_kernel[0] * wg_sigs[idx + 3] +
        conv_kernel[1] * wg_sigs[idx + 4] +
        conv_kernel[2] * wg_sigs[idx + 5] +
        conv_kernel[3] * wg_sigs[idx + 6]
    );
}

__kernel void colour(__global float* input, __write_only image2d_t output)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    int2 coord; coord.x = x; coord.y = y;
    uint idx = (uint)(clamp(input[y*512 + x], 0.0f, 1.0f) * 255.0f);
    float4 colour;
    colour.x = colour_lut[idx * 3    ];
    colour.y = colour_lut[idx * 3 + 1];
    colour.z = colour_lut[idx * 3 + 2];
    colour.w = 1.0f;
    write_imagef(output, coord, colour);
}
