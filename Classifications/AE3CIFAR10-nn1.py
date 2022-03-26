
import keras
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from keras.datasets import cifar10
from matplotlib import pyplot as plt
import numpy as np
import gzip
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import (
    Dropout,Reshape,Conv2DTranspose,BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Conv2D,UpSampling2D,Input
)

from keras.datasets import fashion_mnist

from keras.models import Model
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
from PIL import Image

def encoder(input_img):

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(64,kernel_size=3,strides=2,padding='same',activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv3 = MaxPooling2D((2,2))(conv2)
    conv4 = Flatten()(conv3)
    conv4 = Dense(1000)(conv4)
    return conv4

def decoder(encode):
    #decoder
    conv5 = Dense(4096)(encode)
    conv6 = Reshape((8, 8, 64))(conv5)
    conv7 = UpSampling2D()(conv6)
    conv8 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(conv7)
    conv9 = BatchNormalization()(conv8)
    conv9 =UpSampling2D()(conv9)
    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    decoded = Conv2D(3, (3, 3), activation='sigmoid',padding='same')(conv10)
    return decoded

def fc(encode):

    l1 = Dropout(0.2)(encode)
    den = Dense(100, activation = 'relu')(l1)
    out = Dense(10, activation = 'softmax')(den)
    return out


def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test

epochs = 50
batch_size = 32
inChannel = 3
x, y = 32, 32
input_img = Input(shape = (x, y, inChannel))
num_classes = 10

autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(loss='mean_squared_error', optimizer = 'adam',metrics = ['accuracy'])
autoencoder.load_weights('autoencoder5.h5')
encode = encoder(input_img)
x_train,y_train,x_test,y_test = load_data()
x_decoded = autoencoder.predict(x_test)

full = Model(input_img, fc(encode))
print(full.summary())

for l1,l2 in zip(full.layers[:8], autoencoder.layers[0:8]):
    l1.set_weights(l2.get_weights())



print(autoencoder.get_weights()[0][1])
print("Stop")
print(full.get_weights()[0][1])

for layer in full.layers[0:8]:
    layer.trainable = False

full.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range = 0.1, height_shift_range = 0.1,horizontal_flip = True)
train_generator = data_generator.flow(x_train,y_train,batch_size)
steps_per_epoch = x_train.shape[0] // batch_size
full.fit(train_generator,epochs = epochs,steps_per_epoch = steps_per_epoch,validation_data=(x_test,y_test),verbose=1)

'''
Epoch 1/50
1562/1562 [==============================] - 62s 39ms/step - loss: 2.0440 - accuracy: 0.3844 - val_loss: 1.4139 - val_accuracy: 0.4997
Epoch 2/50
1562/1562 [==============================] - 64s 41ms/step - loss: 1.4845 - accuracy: 0.4723 - val_loss: 1.2885 - val_accuracy: 0.5391
Epoch 3/50
1562/1562 [==============================] - 61s 39ms/step - loss: 1.4149 - accuracy: 0.4985 - val_loss: 1.2540 - val_accuracy: 0.5523
Epoch 4/50
1562/1562 [==============================] - 61s 39ms/step - loss: 1.3761 - accuracy: 0.5124 - val_loss: 1.2401 - val_accuracy: 0.5587
Epoch 5/50
1562/1562 [==============================] - 61s 39ms/step - loss: 1.3578 - accuracy: 0.5188 - val_loss: 1.2314 - val_accuracy: 0.5626
Epoch 6/50
1562/1562 [==============================] - 61s 39ms/step - loss: 1.3477 - accuracy: 0.5249 - val_loss: 1.2065 - val_accuracy: 0.5701
Epoch 7/50
1562/1562 [==============================] - 61s 39ms/step - loss: 1.3288 - accuracy: 0.5305 - val_loss: 1.1886 - val_accuracy: 0.5818
Epoch 8/50
1562/1562 [==============================] - 61s 39ms/step - loss: 1.3178 - accuracy: 0.5363 - val_loss: 1.2031 - val_accuracy: 0.5788
Epoch 9/50
1562/1562 [==============================] - 61s 39ms/step - loss: 1.3123 - accuracy: 0.5409 - val_loss: 1.1798 - val_accuracy: 0.5831
Epoch 10/50
1562/1562 [==============================] - 60s 39ms/step - loss: 1.3097 - accuracy: 0.5370 - val_loss: 1.1762 - val_accuracy: 0.5917
Epoch 11/50
1562/1562 [==============================] - 60s 38ms/step - loss: 1.3081 - accuracy: 0.5404 - val_loss: 1.1793 - val_accuracy: 0.5937
Epoch 12/50
1562/1562 [==============================] - 60s 39ms/step - loss: 1.2972 - accuracy: 0.5462 - val_loss: 1.1812 - val_accuracy: 0.5907
Epoch 13/50
1562/1562 [==============================] - 61s 39ms/step - loss: 1.2949 - accuracy: 0.5447 - val_loss: 1.1601 - val_accuracy: 0.5981
Epoch 14/50
1562/1562 [==============================] - 60s 38ms/step - loss: 1.2922 - accuracy: 0.5458 - val_loss: 1.1452 - val_accuracy: 0.6035
Epoch 15/50
1562/1562 [==============================] - 61s 39ms/step - loss: 1.2919 - accuracy: 0.5481 - val_loss: 1.1867 - val_accuracy: 0.5839
Epoch 16/50
1562/1562 [==============================] - 60s 39ms/step - loss: 1.2839 - accuracy: 0.5509 - val_loss: 1.2025 - val_accuracy: 0.5832
Epoch 17/50
1562/1562 [==============================] - 61s 39ms/step - loss: 1.2807 - accuracy: 0.5505 - val_loss: 1.1795 - val_accuracy: 0.5850
Epoch 18/50
1562/1562 [==============================] - 63s 40ms/step - loss: 1.2838 - accuracy: 0.5483 - val_loss: 1.1708 - val_accuracy: 0.5907
Epoch 19/50
1562/1562 [==============================] - 60s 39ms/step - loss: 1.2723 - accuracy: 0.5572 - val_loss: 1.1516 - val_accuracy: 0.5973
Epoch 20/50
1562/1562 [==============================] - 60s 38ms/step - loss: 1.2748 - accuracy: 0.5567 - val_loss: 1.1463 - val_accuracy: 0.6003
Epoch 21/50
1562/1562 [==============================] - 60s 38ms/step - loss: 1.2823 - accuracy: 0.5529 - val_loss: 1.1420 - val_accuracy: 0.5957
Epoch 22/50
1562/1562 [==============================] - 60s 38ms/step - loss: 1.2750 - accuracy: 0.5540 - val_loss: 1.1681 - val_accuracy: 0.5920
Epoch 23/50
1562/1562 [==============================] - 62s 39ms/step - loss: 1.2703 - accuracy: 0.5540 - val_loss: 1.1505 - val_accuracy: 0.6001
Epoch 24/50
1562/1562 [==============================] - 66s 42ms/step - loss: 1.2675 - accuracy: 0.5565 - val_loss: 1.1438 - val_accuracy: 0.6032
Epoch 25/50
1562/1562 [==============================] - 60s 38ms/step - loss: 1.2712 - accuracy: 0.5555 - val_loss: 1.1810 - val_accuracy: 0.5901
Epoch 26/50
1562/1562 [==============================] - 60s 38ms/step - loss: 1.2685 - accuracy: 0.5571 - val_loss: 1.1687 - val_accuracy: 0.5951
Epoch 27/50
1562/1562 [==============================] - 64s 41ms/step - loss: 1.2690 - accuracy: 0.5552 - val_loss: 1.1601 - val_accuracy: 0.5990
Epoch 28/50
1562/1562 [==============================] - 62s 39ms/step - loss: 1.2652 - accuracy: 0.5573 - val_loss: 1.1309 - val_accuracy: 0.6066
Epoch 29/50
1562/1562 [==============================] - 63s 41ms/step - loss: 1.2715 - accuracy: 0.5573 - val_loss: 1.1585 - val_accuracy: 0.5978
Epoch 30/50
1562/1562 [==============================] - 54s 35ms/step - loss: 1.2622 - accuracy: 0.5604 - val_loss: 1.1345 - val_accuracy: 0.6055
Epoch 31/50
1562/1562 [==============================] - 53s 34ms/step - loss: 1.2617 - accuracy: 0.5608 - val_loss: 1.1463 - val_accuracy: 0.5969
Epoch 32/50
1562/1562 [==============================] - 58s 37ms/step - loss: 1.2648 - accuracy: 0.5574 - val_loss: 1.1788 - val_accuracy: 0.5873
Epoch 33/50
1562/1562 [==============================] - 54s 35ms/step - loss: 1.2624 - accuracy: 0.5590 - val_loss: 1.1379 - val_accuracy: 0.6053
Epoch 34/50
1562/1562 [==============================] - 53s 34ms/step - loss: 1.2591 - accuracy: 0.5601 - val_loss: 1.1550 - val_accuracy: 0.5978
Epoch 35/50
1562/1562 [==============================] - 53s 34ms/step - loss: 1.2597 - accuracy: 0.5624 - val_loss: 1.1285 - val_accuracy: 0.6062
Epoch 36/50
1562/1562 [==============================] - 53s 34ms/step - loss: 1.2556 - accuracy: 0.5630 - val_loss: 1.1400 - val_accuracy: 0.6071
Epoch 37/50
1562/1562 [==============================] - 53s 34ms/step - loss: 1.2588 - accuracy: 0.5597 - val_loss: 1.1398 - val_accuracy: 0.6063
Epoch 38/50
1562/1562 [==============================] - 53s 34ms/step - loss: 1.2572 - accuracy: 0.5631 - val_loss: 1.1516 - val_accuracy: 0.5981
Epoch 39/50
1562/1562 [==============================] - 53s 34ms/step - loss: 1.2615 - accuracy: 0.5585 - val_loss: 1.1425 - val_accuracy: 0.6058
Epoch 40/50
1562/1562 [==============================] - 53s 34ms/step - loss: 1.2589 - accuracy: 0.5639 - val_loss: 1.1382 - val_accuracy: 0.6055
Epoch 41/50
1562/1562 [==============================] - 53s 34ms/step - loss: 1.2576 - accuracy: 0.5621 - val_loss: 1.1589 - val_accuracy: 0.5955
Epoch 42/50
1562/1562 [==============================] - 53s 34ms/step - loss: 1.2610 - accuracy: 0.5603 - val_loss: 1.1528 - val_accuracy: 0.6034
Epoch 43/50
1562/1562 [==============================] - 53s 34ms/step - loss: 1.2531 - accuracy: 0.5636 - val_loss: 1.1366 - val_accuracy: 0.6063
Epoch 44/50
1562/1562 [==============================] - 116s 74ms/step - loss: 1.2526 - accuracy: 0.5641 - val_loss: 1.1511 - val_accuracy: 0.6020
Epoch 45/50
1562/1562 [==============================] - 54s 35ms/step - loss: 1.2568 - accuracy: 0.5607 - val_loss: 1.1468 - val_accuracy: 0.6053
Epoch 46/50
1562/1562 [==============================] - 53s 34ms/step - loss: 1.2498 - accuracy: 0.5651 - val_loss: 1.1542 - val_accuracy: 0.6000
Epoch 47/50
1562/1562 [==============================] - 53s 34ms/step - loss: 1.2582 - accuracy: 0.5642 - val_loss: 1.1151 - val_accuracy: 0.6127
Epoch 48/50
1562/1562 [==============================] - 54s 34ms/step - loss: 1.2582 - accuracy: 0.5599 - val_loss: 1.1391 - val_accuracy: 0.6048
Epoch 49/50
1562/1562 [==============================] - 54s 34ms/step - loss: 1.2486 - accuracy: 0.5647 - val_loss: 1.1603 - val_accuracy: 0.5957
Epoch 50/50
1562/1562 [==============================] - 53s 34ms/step - loss: 1.2560 - accuracy: 0.5642 - val_loss: 1.1352 - val_accuracy: 0.6049

'''
