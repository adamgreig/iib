import yaml
import pickle
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


class gkde:
    def train(self, positive, negative, validate):
        self.pdf = scipy.stats.gaussian_kde(positive.T)

    def score(self, x):
        return np.log(self.pdf(x))


class gmm:
    def train(self, positive, negative, validate):
        self.clf = mixture.GMM(n_components=1, covariance_type='full',
                               n_iter=2000, n_init=500)
        self.clf.fit(positive)

    def score(self, x):
        return self.clf.score(x)


class ocsvm:
    def train(self, positive, negative, validate):
        self.scaler = preprocessing.StandardScaler().fit(positive)
        X = self.scaler.transform(positive)
        self.clf = svm.OneClassSVM(nu=0.1, gamma=0.1, cache_size=512)
        self.clf.fit(X)

    def score(self, x):
        d = self.clf.decision_function(self.scaler.transform(x))
        d[d > 0] = 0.0
        d[d < 0] *= 100.0
        return d


class rf:
    def train(self, positive, negative, validate):
        X = np.vstack((positive, negative))
        y = [1]*positive.shape[0] + [0]*negative.shape[0]
        self.clf = ensemble.RandomForestClassifier(n_estimators=1000)
        self.clf.fit(X, y)

    def score(self, x):
        return self.clf.predict_log_proba(x)[:, 1]


models = [
    ["(Train all and save results)"],
    ["Gaussian Kernel Density Estimate", gkde],
    ["Gaussian Mixture Model", gmm],
    ["One-Class Support Vector Machine", ocsvm],
    ["Random Forest", rf]
]


def learn(model):
    print("Loading feature data...")
    positive = np.load("corpus/patterns.npy")
    negative = np.load("corpus/nopatterns.npy")
    validate = np.load("corpus/validation.npy")
    print("Fitting model...")
    model.train(positive, negative, validate)


def score(model):
    print("Computing scores on examples...")
    with open("corpus/manifest.yaml") as f:
        manifest = yaml.load(f)
    scores = []
    for cls in manifest.keys():
        for img in sorted(manifest[cls].keys()):
            path = "corpus/"+manifest[cls][img]["path"]
            x = features.stats(path)
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
        with open("corpus/trained_models.p", "wb") as f:
            pickle.dump(trained_models, f)

if __name__ == "__main__":
    main()
