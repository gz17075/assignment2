import keras
import pickle
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import pyplot
from keras.layers import UpSampling2D
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D




"""
=======================================================================
This code is citing the website
The title of page: The Keras Blog
Author: Francois Chollet
Date: Sat 14 May 2016
URL: https://blog.keras.io/building-autoencoders-in-keras.html
Access date: Tue 24 April 2018
=======================================================================
"""




# the data, shuffled and split between train and test sets.
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#Open shuffled datasets.
with open('x1_train_mnist.pkl', 'rb') as f:
    x1_train = pickle.load(f)

with open('x2_train_mnist.pkl', 'rb') as f:
    x2_train = pickle.load(f)

with open('x3_train_mnist.pkl', 'rb') as f:
    x3_train = pickle.load(f)




#Encoder
input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format


x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)



#Decoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)




autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')



x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


x1_train = x1_train.astype('float32') / 255.
x1_train = np.reshape(x1_train, (len(x1_train), 28, 28, 1))

x2_train = x2_train.astype('float32') / 255.
x2_train = np.reshape(x2_train, (len(x2_train), 28, 28, 1))

x3_train = x3_train.astype('float32') / 255.
x3_train = np.reshape(x3_train, (len(x3_train), 28, 28, 1))


#Train the model.
autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))



decoded_imgs = autoencoder.predict(x_test)


#Show original and reconstructed pictures.

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n )
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



print("==============================================================")
print("Epochs is : 20")


print("==============================================================")
score_test = autoencoder.evaluate(x_test, x_test, verbose=0)
accuracy_test = 1 - score_test
print("The loss on x_test : %.12f" %score_test)
print("The accuracy on x_test : %.12f" %accuracy_test)
print("==============================================================")


score_x_train = autoencoder.evaluate(x_train, x_train, verbose=0)
accuracy_x_train = 1 - score_x_train
print("The loss on x_train : %.12f" %score_x_train)
print("The accuracy on x_train : %.12f" %accuracy_x_train)
print("==============================================================")


score_x1_train = autoencoder.evaluate(x1_train, x1_train, verbose=0)
accuracy_x1_train = 1 - score_x1_train
print("The loss on x1_train : %.12f" %score_x1_train)
print("The accuracy on x1_train : %.12f" %accuracy_x1_train)
print("==============================================================")


score_x2_train = autoencoder.evaluate(x2_train, x2_train, verbose=0)
accuracy_x2_train = 1 - score_x2_train
print("The loss on x2_train : %.12f" %score_x2_train)
print("The accuracy on x2_train : %.12f" %accuracy_x2_train)
print("==============================================================")


score_x3_train = autoencoder.evaluate(x3_train, x3_train, verbose=0)
accuracy_x3_train = 1 - score_x3_train
print("The loss on x3_train : %.12f" %score_x3_train)
print("The accuracy on x3_train : %.12f" %accuracy_x3_train)




























