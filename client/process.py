from iib.simulation import run
from iib.scoring import score

import copy


def process(config):
    signals = run.run_simulation(config)
    scores = score.score(signals)
    return scores


def process_genome(genome):
    cfg = copy.copy(standard_config)
    cfg["genome"] = genome
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
    print(process_genome("+A303+A563-62A2"))
