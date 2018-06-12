import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from models import simple_cnn_model

seed = 7
np.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
shape = (X_train.shape[1], X_train.shape[2], 1)

# flatten images to vector
X_train = X_train.reshape(X_train.shape[0], *shape).astype("float32")
X_test = X_test.reshape(X_test.shape[0], *shape).astype("float32")

# normalize inputs
X_train = X_train/255
X_test = X_test/255

# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]

model = simple_cnn_model(shape, num_classes)
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=10,
          batch_size=200,
          verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))
# save it
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
