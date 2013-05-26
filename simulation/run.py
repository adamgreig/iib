from iib.simulation import colour, genome, features, reduction, convolution
from iib.simulation import CLContext

import sys
import os.path
import numpy as np
import pyopencl as cl


def set_up_signals(config):
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

    return sigs


def build_programs(clctx, config):
    sigmas = [s['diffusion'] for s in config['signals'][:4]]
    colony = genome.genome_cl(config['genome'])
    colony += convolution.gaussian_cl(sigmas)
    colony = cl.Program(clctx.ctx, colony).build()
    feats = cl.Program(clctx.ctx, features.features_cl()).build()
    sigma2, sigma4 = [np.sqrt(2.0)] * 4, [2.0] * 4
    blurs2 = cl.Program(clctx.ctx, convolution.gaussian_cl(sigma2)).build()
    blurs4 = cl.Program(clctx.ctx, convolution.gaussian_cl(sigma4)).build()
    rdctns = cl.Program(clctx.ctx, reduction.reduction_sum_cl()).build()
    colours = cl.Program(clctx.ctx, colour.colour_cl()).build()
    return colony, feats, blurs2, blurs4, rdctns, colours


def dump_image(clctx, colours, ibuf_1a, ibuf_1b, s, iteration, prefix=None):
    ibuf, ls = (ibuf_1a, s) if s < 4 else (ibuf_1b, s - 4)
    fpath = os.path.dirname(os.path.abspath(__file__))
    sstr = "{0:X}".format(s)
    if prefix:
        sstr = "{0}_{1}".format(prefix, sstr)
    path = fpath + "/output/{0}_{1:05d}.png".format(sstr, iteration)
    colour.dump_colour_image(clctx, colours, ibuf, ls, path)


def run_simulation(config):
    if len(sys.argv) > 1:
        platform_idx = int(sys.argv[1])
        device_idx = int(sys.argv[2])
        dev = cl.get_platforms()[platform_idx].get_devices()[device_idx]
        ctx = cl.Context(devices=[dev])
    else:
        ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    ifmt_f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
    gs = config['grid_size']
    wgs = 16
    clctx = CLContext(ctx, queue, ifmt_f, gs, wgs)
    sigs = set_up_signals(config)
    colony, feats, blurs2, blurs4, rdctns, colours = build_programs(clctx,
                                                                    config)

    sigs_a = sigs[:, :, :4].reshape(gs*gs*4)
    sigs_b = sigs[:, :, 4:].reshape(gs*gs*4)
    ifmt_f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
    ibuf_1a = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    ibuf_1b = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    ibuf_2a = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    ibuf_2b = cl.Image(ctx, mf.READ_WRITE, ifmt_f, (gs, gs))
    cl.enqueue_copy(queue, ibuf_1a, sigs_a, origin=(0, 0), region=(gs, gs))
    cl.enqueue_copy(queue, ibuf_1b, sigs_b, origin=(0, 0), region=(gs, gs))

    for i in config.get('dump_images', []):
        dump_image(clctx, colours, ibuf_1a, ibuf_1b, i, 0)

    n_iters = config['iterations']
    feature_v = np.zeros((n_iters, 8, 4), np.float32)
    for iteration in range(n_iters):
        colony.genome(queue, (gs, gs), (wgs, wgs),
                      ibuf_1a, ibuf_1b, ibuf_2a, ibuf_2b)
        colony.convolve_x(queue, (gs, gs), (wgs, wgs), ibuf_2a, ibuf_1a)
        colony.convolve_y(queue, (gs, gs), (wgs, wgs), ibuf_1a, ibuf_2a)
        ibuf_1a, ibuf_2a = ibuf_2a, ibuf_1a
        ibuf_1b, ibuf_2b = ibuf_2b, ibuf_1b

        feature_v[iteration] = features.get_features(
            clctx, feats, rdctns, blurs2, blurs4, ibuf_1b)

        for i in config.get('dump_images', []):
            dump_image(clctx, colours, ibuf_1a, ibuf_1b, i, iteration+1)

        if config.get('early_stop') and iteration > 1:
            m = np.max(feature_v[iteration])
            if m < 0.1:
                break

    if config.get('dump_final_image'):
        for i in genome.get_used_genes(config["genome"]):
            dump_image(clctx, colours, ibuf_1a, ibuf_1b,
                       i, iteration+1, config["genome"])

    return iteration, feature_v


test_config = {
    "grid_size": 256,
    "iterations": 100,
    "genome": "+4303+4513-1242",
    "early_stop": True,
    "signals": [
        {"diffusion": 1.0, "initial": "random_float", "initial_scale": 0.2},
        {"diffusion": 1.0, "initial": "random_float", "initial_scale": 0.2},
        {"diffusion": 3.0, "initial": "random_float", "initial_scale": 0.2},
        {"diffusion": 5.0, "initial": "random_float", "initial_scale": 0.2},
        {"initial": 0.0},
        {"initial": 0.0},
        {"initial": 0.0},
        {"initial": 0.0},
    ],
    "dump_images": [0, 1, 4]
}


def profile():
    import time
    from copy import deepcopy
    cfg = deepcopy(test_config)
    cfg['early_stop'] = False
    cfg['dump_images'] = []
    iters = 30
    t0 = time.time()
    for i in range(iters):
        run_simulation(cfg)
    print((time.time() - t0)/iters)


def test():
    import matplotlib.pyplot as plt
    print("Running simulation...")
    itercount, fvs = run_simulation(test_config)
    fvs = fvs[:itercount, :, 0]
    print("Ran for {0} iterations.".format(itercount))
    names = ["edges", "blobs", "variance", "entropy"]
    blobs = np.sum(fvs[:, 1:6], axis=1)
    fvs = np.vstack((fvs[:, 0], blobs, fvs[:, 6], fvs[:, 7])).T
    for i in range(4):
        fvs[:, i] /= np.max(fvs[:, i])
        plt.plot(fvs[:, i], label=names[i])
    plt.legend()
    plt.title("Feature Vector Evolution ({0})".format(test_config['genome']))
    plt.ylabel("Normalised Value")
    plt.xlabel("Iteration")
    plt.show()

if __name__ == "__main__":
    #profile()
    test()
