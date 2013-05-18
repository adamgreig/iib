import yaml
import features
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn import mixture
from skimage import io


cls_lut = {
    "patterns": "b",
    "nopatterns": "r",
    "validation": "g"
}


def gkde_train(positive, negative, validate):
    return scipy.stats.gaussian_kde(positive.T)


def gkde_score(model, x):
    return np.log(model(x))[0]


def gmm_train(positive, negative, validate):
    clf = mixture.GMM(n_components=3, covariance_type='full', n_iter=2000,
                      n_init=500)
    clf.fit(positive)
    return clf


def gmm_score(model, x):
    return int(model.score(x)[0])


models = {
    "Gaussian Kernel Density Estimate": [gkde_train, gkde_score],
    "Gaussian Mixture Model": [gmm_train, gmm_score]
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


def main():
    print("Select model:")
    for i, k in enumerate(models.keys()):
        print("[{0}] {1}".format(i+1, k))
    m = int(input("> ")) - 1
    if not 0 <= m <= len(models.keys()):
        print("Invalid selection.")
        return
    k = list(models.keys())[m]
    learn(models[k][0], models[k][1], k)


if __name__ == "__main__":
    main()
