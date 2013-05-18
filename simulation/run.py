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
    gs, wgs = config['grid_size'], config['wg_size']
    sigs = np.empty((gs, gs, 16), np.float32)

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

    sigmas = [s['diffusion'] for s in config['signals']]
    progstr = genome.genome_cl(config['genome'])
    progstr += diffusion.diffusion_cl(sigmas, wgs)
    if config.get('dump_images'):
        progstr += colour.colour_cl()
    program = cl.Program(ctx, progstr).build()

    sigs = sigs.reshape(gs*gs*16)
    buf_a = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=sigs)
    buf_b = cl.Buffer(ctx, mf.READ_WRITE, size=4*16*gs*gs)

    if config.get('dump_images'):
        image = np.empty(gs*gs*4, np.uint8)
        c_order, c_type = cl.channel_order.RGBA, cl.channel_type.UNORM_INT8
        ifmt = cl.ImageFormat(c_order, c_type)
        ibuf = cl.Image(ctx, mf.WRITE_ONLY, ifmt, (gs, gs), None)

        def dump_image(s, iteration):
            bsig = str(chr(s)).encode()
            program.colour(queue, (gs, gs), None, buf_a, bsig, ibuf)
            cl.enqueue_copy(
                queue, image, ibuf, origin=(0, 0), region=(gs, gs)).wait()
            img = Image.fromarray(image.reshape((gs, gs, 4)))
            fpath = os.path.dirname(os.path.abspath(__file__))
            path = fpath + "/output/{0}_{1:05d}.png".format(s, iteration)
            img.save(path)

        for i in config.get('dump_images'):
            dump_image(i, 0)

    n_iters = config['iterations']
    for iteration in range(n_iters):
        program.genome(queue, (gs, gs), None, buf_a, buf_b)
        program.diffusion(queue, (gs, gs), (wgs, 1), buf_b, buf_a)
        program.diffusion(queue, (gs, gs), (1, wgs), buf_a, buf_b)
        buf_a, buf_b = buf_b, buf_a

        if config.get('dump_images'):
            for i in config.get('dump_images'):
                dump_image(i, iteration+1)

    cl.enqueue_copy(queue, sigs, buf_a).wait()
    return sigs.reshape((gs, gs, 16))


test_config = {
    "grid_size": 512,
    "wg_size": 128,
    "iterations": 100,
    "genome": "+A303+A513-12A2",
    "signals": [
        {"diffusion": 1.5, "initial": 0.0},
        {"diffusion": 3.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": "random_binary"},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0}
    ],
    "dump_images": [0, 1, 10]
}


def main():
    run_simulation(test_config)

if __name__ == "__main__":
    main()
