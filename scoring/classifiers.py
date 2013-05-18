import numpy as np
import scipy.stats
from sklearn import mixture, svm, preprocessing, ensemble


class gkde:
    def train(self, positive, negative, validate):
        self.pdf = scipy.stats.gaussian_kde(positive.T)

    def score(self, x):
        return np.log(self.pdf(x.T))


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
