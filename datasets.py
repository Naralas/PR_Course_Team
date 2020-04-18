import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

paths = {
    "csv-train": "data/mnist-csv-format/mnist_train.csv",
    "csv-test": "data/mnist-csv-format/mnist_test.csv",
    "png-train": "data/mnist-png-format/train/",
    "png-test": "data/mnist-png-format/test/",
    "png-val": "data/mnist-png-format/val/",
    "png-perm-train": "data/mnist-permutated-png-format/train/",
    "png-perm-test": "data/mnist-permutated-png-format/test/",
    "png-perm-val": "data/mnist-permutated-png-format/val/",
}


def read_csv(file, N=None):
    data = np.loadtxt(file, delimiter=",", dtype=int, max_rows=N)
    x, y = data[:, 1:], data[:, 0]
    return x, y


def read_png(folder):
    x = []
    y = []
    classes_folders = os.listdir(folder)
    class_ = [int(f) for f in classes_folders]
    classes_folders = [os.path.join(folder, f) for f in classes_folders]
    for classes_folder, y_ in zip(classes_folders, class_):
        files = os.listdir(classes_folder)
        files = [os.path.join(classes_folder, f) for f in files]
        for file in files:
            img = plt.imread(file)
            img = rgb2gray(img)
            img = img.reshape((-1))
            # img = img * 255
            # img = img.astype(int)
            x.append(img)
            y.append(y_)

    return x, y
