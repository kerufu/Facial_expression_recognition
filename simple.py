from model import custom_conv2d, custom_dense
import setting
import dataset_worker


import tensorflow as tf
import cv2
import numpy as np

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model = tf.keras.Sequential([
    custom_conv2d(64, 3),
    custom_conv2d(128, 3),
    custom_conv2d(256, 2),
    custom_conv2d(512, 2),
    tf.keras.layers.Flatten(),
    custom_dense(1024),
    tf.keras.layers.Dense(setting.feature_size, activity_regularizer=tf.keras.regularizers.L1L2()),
    custom_dense(256),
            custom_dense(128),
            custom_dense(64),
            custom_dense(32),
            tf.keras.layers.Dense(setting.num_classes)
])
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