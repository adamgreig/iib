from iib.scoring.features import corpus_path, stats

import pickle
import numpy as np
from skimage.transform import resize


trained_models = None


def preload_models():
    global trained_models
    with open(corpus_path("trained_models.p"), "rb") as f:
        trained_models = pickle.load(f)


def score(sigs, sigs_used=list(range(16))):
    global trained_models
    if not trained_models:
        preload_models()
    x = [stats(resize(sigs[:, :, s].clip(0, 1), (64, 64))) for s in sigs_used]
    scores = np.array([m.score(np.array(x)) for m in trained_models])
    mask = np.ones_like(scores).astype(np.bool)
    mask[:, sigs_used] = 0
    return list(np.ma.masked_array(scores, mask).max(axis=1).flatten())


if __name__ == "__main__":
    from skimage import io
    path = input("Enter path to image: ")
    im = io.imread(path, as_grey=True).astype(np.float32) / 255.0
    sigs = np.zeros((64, 64, 16), np.float32)
    sigs[:, :, 0] = im
    print(sigs)
    print("Scores:", score(sigs, [0]))
