import tensorflow as tf

import time

from model import encoder, classifier
import setting
from dataset_worker import dataset_worker
import os
os.chdir("..")

e = encoder()
c = classifier()

c_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.clipnorm)
e_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.clipnorm)

cfce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=setting.soft_label_ratio)

train_metric = tf.keras.metrics.CategoricalAccuracy()
test_metric = tf.keras.metrics.CategoricalAccuracy()

@tf.function
def train_step(batch):
    image, one_hot = batch["data"], batch["one_hot_coding_label"]
    with tf.GradientTape() as tape_e:
        with tf.GradientTape() as tape_c:
            features = e(image, training=True)
            pred = c(features, training=True)

            print(one_hot.shape, pred.shape)

            loss_e = cfce(one_hot, pred)

            loss_c = cfce(one_hot, pred)

    gradient_e = tape_e.gradient(loss_e, e.trainable_variables)
    e_opt.apply_gradients(zip(gradient_e, e.trainable_variables))

    gradient_c = tape_c.gradient(loss_c, c.trainable_variables)
    c_opt.apply_gradients(zip(gradient_c, c.trainable_variables))

    train_metric.update_state(one_hot, pred)

@tf.function
def test_step(batch):
    image, one_hot = batch["data"], batch["one_hot_coding_label"]
    
    features = e(image)
    pred = c(features)

    test_metric.update_state(one_hot, pred)

def train(epoch):
    dw = dataset_worker()

    train_dataset = dw.train_dataset.shuffle(setting.batch_size, reshuffle_each_iteration=True)
    validation_dataset = dw.validation_dataset.shuffle(setting.batch_size, reshuffle_each_iteration=True)
    for epoch_num in range(epoch):
        start = time.time()

        train_metric.reset_state()
        test_metric.reset_state()
        
        for batch in train_dataset:
            train_step(batch)
        for batch in validation_dataset:
            test_step(batch)
       
        print('Time for epoch {} is {} sec'.format(epoch_num + 1, time.time()-start), 'red')

        print("Train Classifier Accuracy: " + str(train_metric.result().numpy()))
        print("Test Classifier Accuracy: " + str(test_metric.result().numpy()))

train(50)

