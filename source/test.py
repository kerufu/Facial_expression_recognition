from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

import tensorflow as tf
import setting


# checkpoint to save best model
from keras.callbacks import ModelCheckpoint

from model import custom_conv2d

batch_size = 128

# size of the image: 48*48 pixels
pic_size = 48

# input path for the images
base_path = "../dataset/"

train_datagen = ImageDataGenerator(rescale=1.0/255.0)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(base_path + "train",
                                                    target_size=(48, 48),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(base_path + "validation",
                                                              target_size=(
                                                                  48, 48),
                                                              color_mode="grayscale",
                                                              batch_size=batch_size,
                                                              class_mode='categorical',
                                                              shuffle=False)


# number of possible label values
nb_classes = 7

model = tf.keras.Sequential([
    custom_conv2d(64, 3),
    custom_conv2d(128, 5),
    custom_conv2d(512, 3),
    custom_conv2d(512, 3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(setting.feature_size, activity_regularizer=tf.keras.regularizers.L1L2()),
    tf.keras.layers.Dense(setting.num_classes)
])

opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# number of epochs to train the NN
epochs = 50

checkpoint = ModelCheckpoint(
    "model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(train_generator,
                    steps_per_epoch=train_generator.n//train_generator.batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.n//validation_generator.batch_size,
                    callbacks=callbacks_list
                    )


