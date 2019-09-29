import sys
import os
import time
from keras import callbacks
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization

epochs = 500

train_data_path = '/content/keras_implementation_small/train'
validation_data_path = '/content/keras_implementation_small/val'

img_width, img_height = 150, 150
batch_size = 16
steps_per_epoch = 100
validation_steps = 300
lr = 0.001

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode ="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 2, 2, border_mode ="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

model.add(Convolution2D(64, 2, 2, border_mode ="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

model.add(Convolution2D(64, 2, 2, border_mode ="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=lr),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    #shear_range=0.2,
    #zoom_range=0.2,
    horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

callback = callbacks.TensorBoard(log_dir='/content/tf-log/', histogram_freq=0)


model.fit_generator(
    train_generator,
    steps_per_epoch= steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[callback]
    #validation_steps=validation_steps
    )

model.save('/content/logs/model.h5')
model.save_weights('/content/logs/weights.h5')
