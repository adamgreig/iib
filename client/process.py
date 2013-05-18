from iib.simulation import run
from iib.scoring import score

#import numpy as np


def process(config):
    signals = run.run_simulation(config)
    scores = score.score(signals, only_highest_signal=True)
    return scores

test_config = {
    "grid_size": 512,
    "wg_size": 128,
    "iterations": 100,
    "genome": "+A303+A563-62A2",
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
        {"diffusion": 0.0, "initial": "random_binary"},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0},
        {"diffusion": 0.0, "initial": 0.0}
    ]
}


def main():
    print(process(test_config))


if __name__ == "__main__":
    main()
