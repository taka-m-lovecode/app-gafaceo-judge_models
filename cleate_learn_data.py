from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes     = ["bezos", "cook", "pichai", "zack"]
num_classes = len(classes)
img_size    = 50

X = []
Y = []

for index, img_class in enumerate(classes):
    photos_dir = "./" + img_class
    files      = glob.glob(photos_dir + "/*.jpeg")

    for i, img_file in enumerate(files):
        image = Image.open(img_file)
        image = image.convert("RGB")
        image = image.resize((img_size, img_size))
        data  = np.asarray(image)

        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)

xy = (X_train, X_test, y_train, y_test)

np.save("./gafa_ceo.npy", xy)




