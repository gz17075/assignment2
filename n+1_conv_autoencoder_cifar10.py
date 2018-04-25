import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import os
import pickle
import numpy as np




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




batch_size = 32
num_classes = 10
epochs = 21



# the data, shuffled and split between train and test sets.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()



# normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


#Open shuffled datasets.
with open('x1_train_cifar.pkl', 'rb') as f:
    x1_train = pickle.load(f)

with open('x2_train_cifar.pkl', 'rb') as f:
    x2_train = pickle.load(f)

with open('x3_train_cifar.pkl', 'rb') as f:
    x3_train = pickle.load(f)




print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')




#Encoder
input_img = Input(shape=(32, 32, 3))
x = Conv2D(64, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)




#Decoder
x = Conv2D(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
decoded = Activation('sigmoid')(x)




autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')



#Train the model.
history = autoencoder.fit(x_train, x_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, x_test),
                    shuffle=True)





decoded_imgs = autoencoder.predict(x_test)





# definition to show original image and reconstructed image
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_train[i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display predicted pictures
    ax = plt.subplot(2, n, i +1 + n )
    plt.imshow(decoded_imgs[i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()





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

