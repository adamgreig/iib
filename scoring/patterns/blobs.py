import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import transform, io, filter, morphology, measure, feature

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
    results = [("File", "N1", "N2", "N3", "N4"),
               ("----", "--", "--", "--", "--")]
    with open("corpus/manifest.yaml") as f:
        manifest = yaml.load(f)
    for img in sorted(manifest.keys()):
        path = "corpus/"+manifest[img]["path"]
        result = blob_stats(path)
        result = "{0} {1} {2} {3} {4} ".format(img, *result)
        results.append(result.split(" "))
    widths = [max(map(len, col)) for col in zip(*results)]
    for row in results:
        print("  ".join((val.ljust(width) for val, width in zip(row, widths))))

if __name__ == "__main__":
    main()
