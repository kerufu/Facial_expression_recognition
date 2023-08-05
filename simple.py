from model import custom_conv2d, custom_dense
import setting
import dataset_worker


import tensorflow as tf
import cv2
import numpy as np

from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
nb_classes = 7
# Initialising the CNN
model = Sequential()

# 1 - Convolution
model.add(Conv2D(64,(3,3), padding='same', strides=2, input_shape=(48, 48,1)))
model.add(BatchNormalization())
model.add(Activation('leaky_relu'))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(128,(5,5), padding='same', strides=2))
model.add(BatchNormalization())
model.add(Activation('leaky_relu'))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(512,(3,3), padding='same', strides=2))
model.add(BatchNormalization())
model.add(Activation('leaky_relu'))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(512,(3,3), padding='same', strides=2))
model.add(BatchNormalization())
model.add(Activation('leaky_relu'))
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('leaky_relu'))
model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('leaky_relu'))
model.add(Dropout(0.25))

model.add(Dense(nb_classes, activation='softmax'))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

epochs = 50

checkpoint = tf.keras.callbacks.ModelCheckpoint("model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

dw = dataset_worker.dataset_worker()
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1.0/255.0)

validation_datagen = ImageDataGenerator(rescale= 1.0/255)
base_path = "dataset/"
train_generator = train_datagen.flow_from_directory(base_path + "train",
                                                    target_size=(48,48),
                                                    color_mode="grayscale",
                                                    batch_size=64,
                                                    class_mode='categorical',
                                                    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(base_path + "validation",
                                                    target_size=(48,48),
                                                    color_mode="grayscale",
                                                    batch_size=64,
                                                    class_mode='categorical',
                                                    shuffle=False)

history = model.fit(train_generator,
                                epochs=epochs,
                                validation_data = validation_generator,
                                callbacks=callbacks_list
                                )