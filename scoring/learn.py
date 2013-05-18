from iib.scoring.features import corpus_path, stats as feature_stats
from iib.scoring.classifiers import gkde, gmm, ocsvm, rf
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

cls_lut = {
    "patterns": "b",
    "nopatterns": "r",
    "validation": "g"
}

models = [
    ["(Train all and save results)"],
    ["Gaussian Kernel Density Estimate", gkde],
    ["Gaussian Mixture Model", gmm],
    ["One-Class Support Vector Machine", ocsvm],
    ["Random Forest", rf]
]


def learn(model):
    print("Loading feature data...")
    positive = np.load(corpus_path("patterns.npy"))
    negative = np.load(corpus_path("nopatterns.npy"))
    validate = np.load(corpus_path("validation.npy"))
    print("Fitting model...")
    model.train(positive, negative, validate)


def score(model):
    print("Computing scores on examples...")
    with open(corpus_path("manifest.yaml")) as f:
        manifest = yaml.load(f)
    scores = []
    for cls in manifest.keys():
        for img in sorted(manifest[cls].keys()):
            path = corpus_path(manifest[cls][img]["path"])
            im = io.imread(path)
            x = feature_stats(im)
            score = float(model.score(np.array([x])))
            scores.append((score, path, cls))
    return scores


def plot(scores, modelname):
    print("Plotting scores and test images...")
    for idx, (score, path, cls) in enumerate(reversed(sorted(scores))):
        img = io.imread(path)
        rows = int(np.ceil(np.sqrt(len(scores))))
        ax = plt.subplot(rows, rows, idx+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        t = plt.title("{0:g}".format(score))
        for s in ("bottom", "top", "left", "right"):
            col = cls_lut.get(cls, 'k')
            ax.spines[s].set_color(col)
            t.set_color(col)

    plt.suptitle("Scores for {0} (blue=training, red=reject, green=validation)"
                 .format(modelname))
    plt.show()


def main():
    print("Select model:")
    for i in range(len(models)):
        print("[{0}] {1}".format(i, models[i][0]))
    m = int(input("> "))
    if not 0 <= m < len(models):
        print("Invalid selection.")
        return
    if m != 0:
        model = models[m][1]()
        learn(model)
        scores = score(model)
        plot(scores, models[m][0])
    else:
        trained_models = []
        for m in models[1:]:
            print("Training", m[0])
            model = m[1]()
            learn(model)
            trained_models.append(model)
        path = corpus_path("trained_models.p")
        print("Saving trained models to", path)
        with open(path, "wb") as f:
            pickle.dump(trained_models, f)

if __name__ == "__main__":
    main()
