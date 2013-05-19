from iib.simulation import run, genome as sim_genome
from iib.scoring import score

import copy


def process(config):
    signals = run.run_simulation(config)
    genes_used = sim_genome.get_used_genes(config["genome"])
    scores = score.score(signals, genes_used)
    return scores


def process_genome(genome):
    cfg = copy.copy(standard_config)
    cfg["genome"] = genome
    return process(cfg)


def rq_job(generation, genome=None, initial=None, config=None):
    if not config:
        if not genome:
            raise RuntimeError("Must provide either a genome or a config.")
        cfg = copy.deepcopy(standard_config)
        cfg["genome"] = genome
        if initial:
            cfg["signals"][10:] = [{"diffusion": 0.0, "initial": initial}]*6
    else:
        cfg = config
    return process(cfg)

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
