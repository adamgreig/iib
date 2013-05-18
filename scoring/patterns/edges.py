import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import transform, io, filter, morphology, measure

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
    return (np.size(stats), np.mean(stats), np.std(stats))

def main():
    results = [("File", "N", "  µ", "  σ"), ("----", "--", "-----", "-----")]
    with open("corpus/manifest.yaml") as f:
        manifest = yaml.load(f)
    for img in sorted(manifest.keys()):
        path = "corpus/"+manifest[img]["path"]
        result = edge_stats(path)
        result = "{0} {1} {2:.2f} {3:.2f}".format(img, *result)
        results.append(result.split(" "))
    widths = [max(map(len, col)) for col in zip(*results)]
    for row in results:
        print("  ".join((val.ljust(width) for val, width in zip(row, widths))))

if __name__ == "__main__":
    main()