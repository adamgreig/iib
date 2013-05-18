from iib.scoring.features import corpus_path, stats as features_stats

import pickle
import numpy as np
from skimage import transform


def score(signals, only_highest_signal=False):
    with open(corpus_path("trained_models.p"), "rb") as f:
        models = pickle.load(f)
    x = []
    for s in range(16):
        im = signals[:, :, s]
        im = transform.resize(im, (64, 64))
        x.append(features_stats(im))
    scores = []
    x = np.array(x)
    for model in models:
        score = model.score(x)
        if only_highest_signal:
            score = np.max(score)
        scores.append(score)
    return np.array(scores)


if __name__ == "__main__":
    sigs = np.zeros((512, 512, 16), np.float32)
    print(score(sigs))
    print(score(sigs, True))
