from iib.simulation import run

import copy


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
    return run.run_simulation(config)

standard_config = {
    "grid_size": 256,
    "iterations": 100,
    "early_stop": True,
    "signals": [
        {"diffusion": 1.0, "initial": 0.0},
        {"diffusion": 1.0, "initial": 0.0},
        {"diffusion": 3.0, "initial": 0.0},
        {"diffusion": 5.0, "initial": 0.0},
        {"initial": "random_float", "initial_scale": 0.2},
        {"initial": "random_float", "initial_scale": 0.2},
        {"initial": "random_float", "initial_scale": 0.2},
        {"initial": "random_float", "initial_scale": 0.2},
    ]
}


if __name__ == "__main__":
    cfg = copy.copy(standard_config)
    cfg["genome"] = "+4303+4513-1242"
    print(run.run_simulation(cfg))
