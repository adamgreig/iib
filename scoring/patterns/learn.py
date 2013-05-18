import yaml
import features
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


cls_lut = {
    "patterns": "b",
    "nopatterns": "r",
    "validation": "g"
}


def learn(train_cb, score_cb, modelname):
    print("Loading feature data...")
    positive = np.load("corpus/patterns.npy")
    negative = np.load("corpus/nopatterns.npy")
    validate = np.load("corpus/validation.npy")
    print("Fitting model...")
    model = train_cb(positive, negative, validate)
    print("Fitted, computing scores on examples...")

    with open("corpus/manifest.yaml") as f:
        manifest = yaml.load(f)

    scored_paths = []
    for cls in manifest.keys():
        for img in sorted(manifest[cls].keys()):
            path = "corpus/"+manifest[cls][img]["path"]
            x = features.edge_stats(path) + features.blob_stats(path)
            score = score_cb(model, np.array(x).reshape((1, 7)))
            scored_paths.append((score, path, cls))

    for idx, (score, path, cls) in enumerate(reversed(sorted(scored_paths))):
        img = io.imread(path)
        rows = int(np.ceil(np.sqrt(len(scored_paths))))
        ax = plt.subplot(rows, rows, idx+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        t = plt.title("{0:.0f}".format(score))
        for s in ("bottom", "top", "left", "right"):
            col = cls_lut.get(cls, 'k')
            ax.spines[s].set_color(col)
            t.set_color(col)

    plt.suptitle("Scores for {0}".format(modelname))
    plt.show()
