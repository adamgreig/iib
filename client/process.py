from iib.simulation import run
from iib.scoring import score

import copy


def process(config):
    signals = run.run_simulation(config)
    genome = config["genome"]
    genes = (genome[i:i+5] for i in range(0, len(genome), 5))
    used = {int(s, 16) for sl in ((g[1], g[3]) for g in genes) for s in sl}
    scores = score.score(signals, list(used))
    return scores


def process_genome(genome):
    cfg = copy.copy(standard_config)
    cfg["genome"] = genome
    return process(cfg)


def rq_job(generation, genome, initial=None, config=None):
    if not config:
        cfg = copy.deepcopy(standard_config)
        cfg["genome"] = genome
        if initial:
            cfg["signals"][10:] = [{"diffusion": 0.0, "initial": initial}]*6
    else:
        cfg = config
    return list(process(cfg).flatten())

standard_config = {
    "grid_size": 512,
    "wg_size": 128,
    "iterations": 100,
    "signals": [
        {"diffusion": 1.0, "initial": 0.0},
        {"diffusion": 1.0, "initial": 0.0},
        {"diffusion": 1.5, "initial": 0.0},
        {"diffusion": 1.5, "initial": 0.0},
        {"diffusion": 2.0, "initial": 0.0},
        {"diffusion": 2.0, "initial": 0.0},
        {"diffusion": 3.0, "initial": 0.0},
        {"diffusion": 3.0, "initial": 0.0},
        {"diffusion": 5.0, "initial": 0.0},
        {"diffusion": 5.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": "random_float"},
        {"diffusion": 0.0, "initial": "random_float"},
        {"diffusion": 0.0, "initial": "random_float"},
        {"diffusion": 0.0, "initial": "random_float"},
        {"diffusion": 0.0, "initial": "random_float"},
        {"diffusion": 0.0, "initial": "random_float"}
    ]
}


if __name__ == "__main__":
    print(process_genome("+A3A3+A563-62A2"))
