from iib.simulation import run, genome

import numpy as np
import copy


def process(config):
    # Run the simulation, get the resulting feature vector evolution
    niters, fvs = run.run_simulation(config)

    # ignore first two FVs due to initial conditions
    fvs = fvs[2:]

    # now find maximum for each feature
    fvs = np.max(fvs, axis=0)

    # sum up all the blob counts
    blobs = np.sum(fvs[1:6], axis=0)

    # reform fvs with iteration count and combined blobs
    niters = np.ones_like(blobs) * niters
    fvs = np.array((niters, fvs[0], blobs, fvs[6], fvs[7]))

    # normalise
    gs = config["grid_size"]
    iters = config["iterations"]
    norms = np.array((iters, (gs**2)/2, (gs**2)/(26*4*5), 0.25, 8.0))
    norms = np.kron(norms, np.ones(4)).reshape(5, 4)
    fvs /= norms

    # weight
    scores = np.dot(config["weights"], fvs)

    # Take the average weight of the genomes in use
    sigs_used = np.array(genome.get_used_genes(config["genome"]))
    sigs_used = sigs_used[sigs_used >= 4] - 4
    if sigs_used.size == 0:
        return 0.0
    mask = np.ones_like(scores).astype(np.bool)
    mask[:, sigs_used] = 0
    score = np.ma.masked_array(scores, mask).mean()

    return float(score)


def rq_job(generation, genome=None, initial=None, config=None):
    if not config:
        if not genome:
            raise RuntimeError("Must provide either a genome or a config.")
        cfg = copy.deepcopy(standard_config)
        cfg["genome"] = genome
        if initial:
            cfg["signals"][4:] = [{"diffusion": 0.0, "initial": initial}]*4
    else:
        cfg = config
    return process(cfg)

standard_config = {
    "grid_size": 256,
    "iterations": 100,
    "early_stop": True,
    "weights": [.3, .2, .3, .1, .1],
    "signals": [
        {"diffusion": 1.0, "initial": "random_float", "initial_scale": 0.2},
        {"diffusion": 1.0, "initial": "random_float", "initial_scale": 0.2},
        {"diffusion": 3.0, "initial": "random_float", "initial_scale": 0.2},
        {"diffusion": 5.0, "initial": "random_float", "initial_scale": 0.2},
        {"initial": 0.0},
        {"initial": 0.0},
        {"initial": 0.0},
        {"initial": 0.0},
    ]
}


if __name__ == "__main__":
    cfg = copy.copy(standard_config)
    cfg["genome"] = "+4303+4513-1242"
    print(process(cfg))
