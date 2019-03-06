from keras.datasets.mnist import load_data
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def load_mnist():
    (X_train, y_train), (X_test, y_test) = load_data()
    return np.expand_dims(X_train, 3) / 255, np.expand_dims(X_test, 3) / 255, np.expand_dims(y_train, 3), np.expand_dims(y_test, 3)

def load_birds(relative_image_path = "../images/", shape = (128, 128)):
    fnames = os.listdir(relative_image_path)
    class_names = []
    for i in range(len(fnames)):
        class_names.append(" ".join(fnames[i].split("_")[0:2]))
    unique_fnames = list(set(class_names))
    class_name_to_class_id = {}
    for i in range(len(unique_fnames)):
        class_name_to_class_id[unique_fnames[i]] = i
    x = np.zeros((len(fnames), *shape))
    y = np.zeros((len(fnames)))
    for i in tqdm(range(len(fnames))):
        x[i] = np.array(Image.open(relative_image_path + fnames[i]).resize(shape).convert("L"))
        y[i] = class_name_to_class_id[class_names[i]]
    return np.array(x), np.array(y)