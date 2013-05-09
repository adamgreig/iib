import yaml
import features
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from skimage import io

def main():
    train = np.load("corpus/patterns.npy")
    reject = np.load("corpus/nopatterns.npy")
    clf = mixture.GMM(n_components=2, covariance_type='full', n_iter=2000,
                      n_init=500)
    print("Fitting model...")
    clf.fit(train)
    print("Fitted, computing scores on examples...")

    with open("corpus/manifest.yaml") as f:
        manifest = yaml.load(f)

    scored_paths = []
    for cls in manifest.keys():
        for img in sorted(manifest[cls].keys()):
            path = "corpus/"+manifest[cls][img]["path"]
            x = features.edge_stats(path) + features.blob_stats(path)
            score = int(clf.score(np.array(x).reshape((1, 7)))[0])
            scored_paths.append((score, path))

    for idx, (score, path) in enumerate(reversed(sorted(scored_paths))):
        img = io.imread(path)
        rows = int(np.ceil(np.sqrt(len(scored_paths))))
        plt.subplot(rows, rows, idx+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(str(score))

    plt.show()
            

if __name__ == "__main__":
    main()
