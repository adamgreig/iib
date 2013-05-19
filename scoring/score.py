from iib.scoring.features import corpus_path, stats

import pickle
import numpy as np
from skimage.transform import resize


def score(signals):
    with open(corpus_path("trained_models.p"), "rb") as f:
        models = pickle.load(f)
    x = [stats(resize(signals[:, :, s], (64, 64))) for s in range(16)]
    scores = np.array([m.score(np.array(x)) for m in models])
    return np.amax(scores, axis=1)


if __name__ == "__main__":
    sigs = np.zeros((512, 512, 16), np.float32)
    print(score(sigs))
