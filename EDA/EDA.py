import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

test_images_folder_path = "/Users/sschickler/ML_DATA/Clouds/test_images"
train_images_folder_path = "/Users/sschickler/ML_DATA/Clouds/train_images"
train_labels_path = "/Users/sschickler/ML_DATA/train.csv"
pickle_file_path = "/Users/sschickler/ML_DATA/train_labels.pickle"


def rleToMask(rleString, height, width):
    rows, cols = height, width
    rleNumbers = np.array(rleString.split(' '), dtype=int)
    rlePairs = rleNumbers.reshape(-1, 2)
    print(rlePairs)
    img = np.zeros(rows * cols, dtype=np.byte)
    for index, length in rlePairs:
        index -= 1
        img[index:index + length] = 255
    img = img.reshape(cols, rows)
    img = img.T
    plt.imshow(img, cmap='gray')
    plt.show()

    return img


def displayImageMaskRLE(rleString, height, width):  # displays image from RLE
    rows, cols = height, width
    rleNumbers = np.array(rleString.split(' '), dtype=int)
    rlePairs = rleNumbers.reshape(-1, 2)
    print(rlePairs)
    img = np.zeros(rows * cols, dtype=np.byte)
    for index, length in rlePairs:
        index -= 1
        img[index:index + length] = 255
    img = img.reshape(cols, rows)
    img = img.T
    plt.imshow(img, cmap='gray')
    plt.show()


labels = pd.read_csv(train_labels_path)
labels.head(10)  # prints out first 10 rows of labels
labels.fillna(-1, inplace=True)
print(labels.head(10))
print(labels.iloc(2)["EncodedPixels"])
rleToMask(labels["EncodedPixels"][0], 1400, 2100)
