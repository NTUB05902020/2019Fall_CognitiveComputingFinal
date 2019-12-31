import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def getLBPFeature(img):
    h, w = img.shape
    # padding image
    #print("original size", img.shape)
    pad_height = 10 * (h // 10 + (h % 10 > 0))
    pad_weight = 10 * (w // 10 + (w % 10 > 0))
    img = np.hstack([img, np.zeros([h, pad_weight-w])])
    img = np.vstack([img, np.zeros([pad_height - h, pad_weight])])
    #print("after resize", img.shape)

    blocks = blockshaped(img, pad_height//10, pad_weight//10)
    #print("cut into", blocks.shape)

    # n_points = 8, radius = 1
    all_hist = []
    for block in blocks:
        LBP = local_binary_pattern(block, 8, 1, method='nri_uniform')
        n_bins = int(LBP.max() + 1)
        # print(block.shape, n_bins)
        hist, bins = np.histogram(LBP.flatten(), bins=n_bins, range=(0,n_bins), density=True)
        hist.resize((59))
        all_hist.append(hist) 

    feature = np.hstack(all_hist)
    return feature

if __name__ == '__main__':
    img = cv2.imread("15.jpg", cv2.IMREAD_GRAYSCALE)
    LBP_Feature = getLBPFeature(img)
    print("LBP Feature: ", LBP_Feature.shape, LBP_Feature)