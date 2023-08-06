import tensorflow as tf

import time

import numpy as np

from model import encoder, classifier
import setting
from dataset_worker import dataset_worker
import os

from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
os.chdir("..")

# Initialising the CNN
model = Sequential()

# 1 - Convolution
model.add(Conv2D(64, (3, 3), padding='same',
          strides=2, input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation('leaky_relu'))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(128, (5, 5), padding='same', strides=2))
model.add(BatchNormalization())
model.add(Activation('leaky_relu'))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(512, (3, 3), padding='same', strides=2))
model.add(BatchNormalization())
model.add(Activation('leaky_relu'))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(512, (3, 3), padding='same', strides=2))
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

model.add(Dense(setting.num_classes))

opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.clipnorm)
cfce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=setting.soft_label_ratio)

train_metric = tf.keras.metrics.CategoricalAccuracy()
test_metric = tf.keras.metrics.CategoricalAccuracy()

# model.compile(optimizer=opt, loss=cfce,
#               metrics=['categorical_accuracy'])

# model.fit(train_data, train_one_hot_coding_label,
#                     epochs=epoch,
#                     validation_data=(test_data, test_one_hot_coding_label),
#                     batch_size=setting.batch_size
#                     )

# @tf.function
def train_step(batch):
    image, one_hot = batch["data"], batch["one_hot_coding_label"]
    with tf.GradientTape() as tape:
        pred = model(image, training=True)
        loss = cfce(one_hot, pred)

    gradient = tape.gradient(loss, model.trainable_variables)
    # tf.print(gradient)
    opt.apply_gradients(zip(gradient, model.trainable_variables))

    train_metric.update_state(one_hot, pred)

# @tf.function
def test_step(batch):
    image, one_hot = batch["data"], batch["one_hot_coding_label"]
    pred = model(image)

    test_metric.update_state(one_hot, pred)
import glob
import cv2
def train(epoch):
    # train_data = []
    # train_one_hot_coding_label = []
    # for ec_index in range(setting.num_classes):
    #     for path in glob.iglob(setting.train_dataset_path+"/"+setting.expression_classes[ec_index]+"/*.jpg"):

    #         img = cv2.imread(path, 0)
    #         train_data.append(img)
            
    #         one_hot_coding = [0] * setting.num_classes
    #         one_hot_coding[ec_index] = 1
    #         train_one_hot_coding_label.append(one_hot_coding)

    # train_data = np.array(train_data)
    # train_data = train_data / 127.5 - 1
    # train_data = train_data[:, :, :, np.newaxis]
    # train_one_hot_coding_label = np.array(train_one_hot_coding_label)

    # test_data = []
    # test_one_hot_coding_label = []
    # for ec_index in range(setting.num_classes):
    #     for path in glob.iglob(setting.validation_dataset_path+"/"+setting.expression_classes[ec_index]+"/*.jpg"):

    #         img = cv2.imread(path, 0)
    #         test_data.append(img)
            
    #         one_hot_coding = [0] * setting.num_classes
    #         one_hot_coding[ec_index] = 1
    #         test_one_hot_coding_label.append(one_hot_coding)

    # test_data = np.array(test_data)
    # test_data = test_data / 127.5 - 1
    # test_data = test_data[:, :, :, np.newaxis]
    # test_one_hot_coding_label = np.array(test_one_hot_coding_label)

    # train_data = tf.convert_to_tensor(train_data)
    # train_one_hot_coding_label = tf.convert_to_tensor(train_one_hot_coding_label)
    # test_data = tf.convert_to_tensor(test_data)
    # test_one_hot_coding_label = tf.convert_to_tensor(test_one_hot_coding_label)

    
    dw = dataset_worker()
    # train_dataset = dw.train_dataset.shuffle(setting.batch_size, reshuffle_each_iteration=True)
    # train_dataset = train_dataset.batch(setting.batch_size)

    # validation_dataset = dw.validation_dataset.shuffle(setting.batch_size, reshuffle_each_iteration=True)
    # validation_dataset = validation_dataset.batch(setting.batch_size)
    
    train_dataset = dw.train_dataset
    validation_dataset = dw.validation_dataset
    
# 
    # train_dataset = tf.data.Dataset.from_tensor_slices({
    #         "data": train_data,
    #         "one_hot_coding_label": train_one_hot_coding_label
    #     })

    # train_dataset = train_dataset.shuffle(setting.batch_size, reshuffle_each_iteration=True)
    # train_dataset = train_dataset.batch(1)


    # validation_dataset = tf.data.Dataset.from_tensor_slices({
    #         "data": test_data,
    #         "one_hot_coding_label": test_one_hot_coding_label
    #     })
    # validation_dataset = validation_dataset.shuffle(setting.batch_size, reshuffle_each_iteration=True)
    # validation_dataset = validation_dataset.batch(1)


    for epoch_num in range(epoch):
        start = time.time()

        validation_dataset = train_dataset.shuffle(validation_dataset.cardinality())
        train_dataset = train_dataset.shuffle(train_dataset.cardinality())

        train_metric.reset_state()
        test_metric.reset_state()

        # indexes = np.random.permutation(len(train_data))
        # train_data = train_data[indexes]
        # train_one_hot_coding_label = train_one_hot_coding_label[indexes]
        # train_dataset = tf.data.Dataset.from_tensor_slices({
        #     "data": train_data,
        #     "one_hot_coding_label": train_one_hot_coding_label
        # }).batch(setting.batch_size)

        # indexes = np.random.permutation(len(test_data))
        # test_data = test_data[indexes]
        # test_one_hot_coding_label = test_one_hot_coding_label[indexes]
        # validation_dataset = tf.data.Dataset.from_tensor_slices({
        #     "data": test_data,
        #     "one_hot_coding_label": test_one_hot_coding_label
        # }).batch(setting.batch_size)
        
        for batch in train_dataset.batch(setting.batch_size):
            train_step(batch)
        for batch in validation_dataset.batch(setting.batch_size):
            test_step(batch)
       
        print('Time for epoch {} is {} sec'.format(epoch_num + 1, time.time()-start), 'red')

        print("Train Classifier Accuracy: " + str(train_metric.result().numpy()))
        print("Test Classifier Accuracy: " + str(test_metric.result().numpy()))

train(50)



