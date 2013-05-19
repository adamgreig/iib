from iib.scoring.features import corpus_path, stats

import pickle
import numpy as np
from skimage.transform import resize


trained_models = None


def preload_models():
    global trained_models
    with open(corpus_path("trained_models.p"), "rb") as f:
        trained_models = pickle.load(f)


def score(signals):
    global trained_models
    if not trained_models:
        preload_models()
    x = [stats(resize(signals[:, :, s], (64, 64))) for s in range(16)]
    scores = np.array([m.score(np.array(x)) for m in trained_models])
    return np.amax(scores, axis=1)


if __name__ == "__main__":
    sigs = np.zeros((512, 512, 16), np.float32)
    print(score(sigs))
