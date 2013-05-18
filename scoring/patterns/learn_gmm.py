import learn
from sklearn import mixture


def train(positive, negative, validate):
    clf = mixture.GMM(n_components=3, covariance_type='full', n_iter=2000,
                      n_init=500)
    clf.fit(positive)
    return clf


def score(model, x):
    return int(model.score(x)[0])

if __name__ == "__main__":
    learn.learn(train, score, "Gaussian Mixture Model")
