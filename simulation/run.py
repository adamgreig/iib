from iib.simulation import colour
from iib.simulation import genome
from iib.simulation import diffusion

import os.path
import numpy as np
import pyopencl as cl

from PIL import Image


def run_simulation(config):
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    gs = config['grid_size']
    sigs = np.empty((gs, gs, 8), np.float32)

    for idx, signal in enumerate(config['signals']):
        if type(signal['initial']) == float:
            sigs[:, :, idx] = signal['initial']
        elif signal['initial'] == 'random_float':
            sigs[:, :, idx] = np.random.random((gs, gs)).astype(np.float32)
        elif signal['initial'] == 'random_binary':
            t = np.float32
            sigs[:, :, idx] = np.random.randint(0, 2, (gs, gs)).astype(t)
        elif signal['initial'] == 'split':
            sigs[0:(gs//2), :, idx] = 1.0
            sigs[(gs//2):, :, idx] = 0.0
        elif signal['initial'] == 'box':
            m = gs//2
            w = signal.get('box_width', 50)
            sigs[:, :, idx] = 0.0
            sigs[m-w:m+w, m-w:m+w, idx] = 1.0
        if 'initial_scale' in signal:
            sigs[:, :, idx] *= signal['initial_scale']
        if 'initial_offset' in signal:
            sigs[:, :, idx] += signal['initial_offset']

    sigmas = [s['diffusion'] for s in config['signals'][:4]]
    progstr = genome.genome_cl(config['genome'])
    progstr += diffusion.diffusion_cl(sigmas)
    if config.get('dump_images') or config.get('dump_final_image'):
        progstr += colour.colour_cl()
    program = cl.Program(ctx, progstr).build()

    sigs_a = sigs[:, :, :4].reshape(gs*gs*4)
    sigs_b = sigs[:, :, 4:].reshape(gs*gs*4)
    ifmt_f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
    ibuf_1a = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    ibuf_1b = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    ibuf_2a = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    ibuf_2b = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    cl.enqueue_copy(queue, ibuf_1a, sigs_a, origin=(0, 0), region=(gs, gs))
    cl.enqueue_copy(queue, ibuf_1b, sigs_b, origin=(0, 0), region=(gs, gs))

    if config.get('dump_images') or config.get('dump_final_image'):
        image_out = np.empty(gs*gs*4, np.uint8)
        c_order, c_type = cl.channel_order.RGBA, cl.channel_type.UNORM_INT8
        ifmt_u = cl.ImageFormat(c_order, c_type)
        ibuf_o = cl.Image(ctx, mf.WRITE_ONLY, ifmt_u, (gs, gs))

        def dump_image(s, iteration, prefix=None):
            if s < 4:
                ibuf, bsig = ibuf_1a, str(chr(s)).encode()
            else:
                ibuf, bsig = ibuf_1b, str(chr(s-4)).encode()
            program.colour(queue, (gs, gs), (16, 16), ibuf, bsig, ibuf_o)
            cl.enqueue_copy(
                queue, image_out, ibuf_o, origin=(0, 0), region=(gs, gs))
            img = Image.fromarray(image_out.reshape((gs, gs, 4)))
            fpath = os.path.dirname(os.path.abspath(__file__))
            s = "{0:X}".format(s)
            if prefix:
                s = "{0}_{1}".format(prefix, s)
            path = fpath + "/output/{0}_{1:05d}.png".format(s, iteration)
            img.save(path)

        if config.get('dump_images'):
            for i in config.get('dump_images'):
                dump_image(i, 0)

    n_iters = config['iterations']
    for iteration in range(n_iters):
        program.genome(queue, (gs, gs), (16, 16),
                       ibuf_1a, ibuf_1b, ibuf_2a, ibuf_2b)
        program.convolve_x(queue, (gs, gs), (16, 16), ibuf_2a, ibuf_1a)
        program.convolve_y(queue, (gs, gs), (16, 16), ibuf_1a, ibuf_2a)
        ibuf_1a, ibuf_2a = ibuf_2a, ibuf_1a
        ibuf_1b, ibuf_2b = ibuf_2b, ibuf_1b

        if config.get('dump_images'):
            for i in config.get('dump_images'):
                dump_image(i, iteration+1)

    if config.get('dump_final_image'):
        for i in genome.get_used_genes(config["genome"]):
            dump_image(i, iteration+1, config["genome"])
    cl.enqueue_copy(queue, sigs_a, ibuf_1a, origin=(0, 0), region=(gs, gs))
    cl.enqueue_copy(queue, sigs_b, ibuf_1b, origin=(0, 0), region=(gs, gs))
    sigs[:, :, :4] = sigs_a.reshape((gs, gs, 4))
    sigs[:, :, 4:] = sigs_b.reshape((gs, gs, 4))
    return sigs


test_config = {
    "grid_size": 256,
    "iterations": 100,
    "genome": "+4303+4513-1242",
    "signals": [
        {"diffusion": 1.0, "initial": 0.0},
        {"diffusion": 1.0, "initial": 0.0},
        {"diffusion": 3.0, "initial": 0.0},
        {"diffusion": 5.0, "initial": 0.0},
        {"initial": "random_binary"},
        {"initial": "random_binary"},
        {"initial": "random_binary"},
        {"initial": "random_binary"},
    ],
    "dump_images": [0, 1, 4]
}


if __name__ == "__main__":
    print(run_simulation(test_config))
