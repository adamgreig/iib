import yaml
import features
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn import mixture, svm, preprocessing, ensemble
from skimage import io


cls_lut = {
    "patterns": "b",
    "nopatterns": "r",
    "validation": "g"
}


def gkde_train(positive, negative, validate):
    return scipy.stats.gaussian_kde(positive.T)


def gkde_score(model, x):
    return np.log(model(x))


def gmm_train(positive, negative, validate):
    clf = mixture.GMM(n_components=3, covariance_type='full', n_iter=2000,
                      n_init=500)
    clf.fit(positive)
    return clf


def gmm_score(model, x):
    return model.score(x)


def ocsvm_train(positive, negative, validate):
    scaler = preprocessing.StandardScaler().fit(positive)
    X = scaler.transform(positive)
    clf = svm.OneClassSVM(nu=0.1, gamma=0.1, cache_size=512)
    clf.fit(X)
    return [scaler, clf]


def ocsvm_score(model, x):
    d = model[1].decision_function(model[0].transform(x))
    d[d > 0] = 0.0
    d[d < 0] *= 100.0
    return d


def rf_train(positive, negative, validate):
    X = np.vstack((positive, negative))
    y = [1]*positive.shape[0] + [0]*negative.shape[0]
    clf = ensemble.RandomForestClassifier()
    return clf.fit(X, y)


def rf_score(model, x):
    return model.predict_log_proba(x)[:, 1]


models = {
    "Gaussian Kernel Density Estimate": [gkde_train, gkde_score],
    "Gaussian Mixture Model": [gmm_train, gmm_score],
    "One-Class Support Vector Machine": [ocsvm_train, ocsvm_score],
    "Random Forest": [rf_train, rf_score]
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
            score = float(score_cb(model, np.array(x).reshape((1, 7))))
            scored_paths.append((score, path, cls))

    for idx, (score, path, cls) in enumerate(reversed(sorted(scored_paths))):
        img = io.imread(path)
        rows = int(np.ceil(np.sqrt(len(scored_paths))))
        ax = plt.subplot(rows, rows, idx+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        t = plt.title("{0:.5g}".format(score))
        for s in ("bottom", "top", "left", "right"):
            col = cls_lut.get(cls, 'k')
            ax.spines[s].set_color(col)
            t.set_color(col)

    plt.suptitle("Scores for {0} (blue=train, red=reject, green=validation)"
                 .format(modelname))
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
