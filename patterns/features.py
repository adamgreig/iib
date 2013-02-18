import yaml
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, io, filter, morphology, measure, feature

def edge_stats(path, show_images=False):
    im = io.imread(path, as_grey=True)
    if im.shape != (64, 64):
        im = transform.resize(im, (64, 64))
    edges = filter.canny(im, sigma=1)
    edges = morphology.label(edges, neighbors=8)
    stats = [s['Area'] for s in measure.regionprops(edges, properties=['Area'])]
    stats = np.array(stats)
    stats = stats[stats>3]
    if show_images:
        plt.imshow(im, interpolation="nearest")
        plt.show()
        plt.imshow(edges, cmap=plt.cm.Paired, interpolation='nearest')
        plt.show()
    return [np.size(stats), np.mean(stats), np.std(stats)]

def blob_stats(path, show_images=False):
    im = io.imread(path, as_grey=True)
    if im.shape != (64, 64):
        im = transform.resize(im, (64, 64))
    pyramid = tuple(transform.pyramid_laplacian(im, mode='nearest'))
    num_blobs = []
    for idx in range(4):
        dist = np.ceil(8.0 / (3.0 * (idx+1)))
        maxms = feature.peak_local_max(pyramid[idx], min_distance=dist)
        num_blobs.append(maxms.shape[0])
        if show_images:
            plt.subplot(4, 2, 2*idx + 1)
            plt.imshow(im, cmap='gray')
            plt.scatter(maxms.T[1] * (2**idx), maxms.T[0] * (2**idx), s=50,c='g')
            plt.subplot(4, 2, 2*idx + 2)
            plt.imshow(pyramid[idx], cmap='gray')
            plt.scatter(maxms.T[1], maxms.T[0], s=50,c='g')
    if show_images:
        plt.show()
    return num_blobs

def main():
    results = []
    with open("corpus/manifest.yaml") as f:
        manifest = yaml.load(f)
    for img in sorted(manifest.keys()):
        path = "corpus/"+manifest[img]["path"]
        result = edge_stats(path) + blob_stats(path)
        results.append(result)
    results = np.array(results)
    np.save("corpus/features.npy", results)
    print("Features saved to corpus/feature.npy")

if __name__ == "__main__":
    main()
