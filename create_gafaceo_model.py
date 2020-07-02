import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.utils import np_utils
import numpy as np

classes     = ["bezos", "cook", "pichai", "zack"]
num_classes = len(classes)
img_size    = 50

# 学習用データをロードし、学習用データとテストデータを分ける
def learn_data_load():
    X_train, X_test, y_train, y_test = np.load("./gafa_ceo.npy", allow_pickle=True)

    X_train = X_train.astype("float") / 256
    X_test  = X_test.astype("float") / 256
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    print("class:", num_classes)

    model = model_train(X_train, y_train)

    model_eval(model, X_test, y_test)


def model_train(X, y):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

    # コールバックの作成
    # 学習が収束したら途中で打ち切る
    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

    model.fit(X, y, batch_size=32, epochs=200, callbacks=([es_cb]))

    model.save("./gafa_ceo_cnn.h5")

    return model


def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose=1)

    print("test Loss:", scores[0])
    print("test Accuracy:", scores[1])


if __name__ == "__main__":
    learn_data_load()




