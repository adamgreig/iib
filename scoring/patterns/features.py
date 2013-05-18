import yaml
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, io, filter, morphology, measure, feature


def edge_stats(path, show_images=False):
    im = io.imread(path, as_grey=True)
    if im.shape != (64, 64):
        im = transform.resize(im, (64, 64))
    edgs = filter.canny(im, sigma=1)
    edgs = morphology.label(edgs, neighbors=8)
    stats = [s['Area'] for s in measure.regionprops(edgs, properties=['Area'])]
    stats = np.array(stats)
    stats = stats[stats > 3]
    if show_images:
        plt.imshow(im, interpolation="nearest")
        plt.show()
        plt.imshow(edgs, cmap=plt.cm.Paired, interpolation='nearest')
        plt.show()
    if not stats.size:
        return [0, 0, 0]
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
        if maxms == []:
            num_blobs.append(0)
            continue
        num_blobs.append(maxms.shape[0])
        if show_images:
            plt.subplot(4, 2, 2*idx + 1)
            plt.imshow(im, cmap='gray')
            plt.scatter(maxms.T[1] * (2**idx), maxms.T[0] * (2**idx),
                        s=50, c='g')
            plt.subplot(4, 2, 2*idx + 2)
            plt.imshow(pyramid[idx], cmap='gray')
            plt.scatter(maxms.T[1], maxms.T[0], s=50, c='g')
    if show_images:
        plt.show()
    return num_blobs


def main():
    print_stats = True
    with open("corpus/manifest.yaml") as f:
        manifest = yaml.load(f)
    for cls in ("patterns", "nopatterns"):
        results = []
        dresults = [("File", "N", "  µ", "  σ", "B1", "B2", "B3", "B4"),
                    ("----", "--", "-----", "-----", "--", "--", "--", "--")]
        for img in sorted(manifest[cls].keys()):
            path = "corpus/"+manifest[cls][img]["path"]
            result = edge_stats(path) + blob_stats(path)
            results.append(result)
            dresult = "{0} {1} {2:.1f} {3:.1f} {4} {5} {6} {7}".format(img,
                                                                       *result)
            dresults.append(dresult.split(" "))
        if print_stats:
            print("Stats for <{0}>:".format(cls))
            widths = [max(map(len, col)) for col in zip(*dresults)]
            for row in dresults:
                print(" ".join((val.ljust(width) for val, width in zip(row,
                      widths))))
        results = np.array(results)
        outpath = "corpus/{0}.npy".format(cls)
        np.save(outpath, results)
        print("Feature vectors saved to {0}".format(outpath), end='\n\n')

if __name__ == "__main__":
    main()
