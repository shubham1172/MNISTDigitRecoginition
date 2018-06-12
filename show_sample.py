from keras.datasets import mnist
import matplotlib.pyplot as plt
from random import randrange

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_plot = 9
total_ = len(X_train)
# plot random figures
for i in range(num_plot):
    plt.subplot(num_plot**0.5, num_plot**0.5, i+1)
    plt.imshow(X_train[randrange(total_)], cmap=plt.get_cmap("gray"))

plt.show()
