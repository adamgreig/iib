import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

def main():
    train = np.load("corpus/patterns.npy")
    reject = np.load("corpus/nopatterns.npy")
    clf = mixture.GMM(n_components=3, covariance_type='full', n_iter=1000,
                      n_init=100)
    print("Fitting model...")
    clf.fit(train)
    pattern_scores = clf.score(train).astype(np.int)
    reject_scores = clf.score(reject).astype(np.int)
    print("Score for training data:", pattern_scores)
    print("Score for reject data:", reject_scores)

    plt.scatter(pattern_scores, np.zeros_like(pattern_scores), c='g')
    plt.scatter(reject_scores, np.zeros_like(reject_scores), c='r')
    plt.show()

if __name__ == "__main__":
    main()
