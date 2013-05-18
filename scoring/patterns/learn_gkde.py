import learn
import numpy as np
import scipy.stats


def train(positive, negative, validate):
    return scipy.stats.gaussian_kde(positive.T)


def score(model, x):
    return np.log(model(x))[0]


if __name__ == "__main__":
    learn.learn(train, score, "Gaussian Kernel Density Estimate")
