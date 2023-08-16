import time

import tensorflow as tf
import cv2
import numpy as np
from termcolor import cprint

import setting
from model import wgan_generator, wgan_discriminator, classifier, WassersteinLoss

class wgan_worker():

    def __init__(self, g_iteration=1, d_iteration=5) -> None:
        self.g_iteration = g_iteration
        self.d_iteration = d_iteration

        self.g = wgan_generator()
        self.d = wgan_discriminator()
        self.c = classifier()

        try:
            self.g.load_weights(setting.wgan_generator_path)
            self.d.load_weights(setting.wgan_discriminator_path)
            self.c.load_weights(setting.classifier_path)
            print("wgan model weight loaded")
        except:
            print("wgan model weight not found")

        self.g_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)
        self.d_opt = tf.keras.optimizers.RMSprop(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.gradient_clip_norm, weight_decay=setting.weight_decay)

        self.wl = WassersteinLoss()
        self.cfce = tf.keras.losses.CategoricalFocalCrossentropy(from_logits=True, label_smoothing=setting.soft_label_ratio)

        self.g_train_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)
        self.d_train_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)
        self.c_train_acc_metric = tf.keras.metrics.CategoricalAccuracy()

        self.g_train_loss_metric = tf.keras.metrics.Mean()
        self.d_train_loss_metric = tf.keras.metrics.Mean()
        self.c_train_loss_metric = tf.keras.metrics.Mean()

        self.g_test_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)
        self.d_test_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)
        self.c_test_acc_metric = tf.keras.metrics.CategoricalAccuracy()

        self.g_test_loss_metric = tf.keras.metrics.Mean()
        self.d_test_loss_metric = tf.keras.metrics.Mean()
        self.c_test_loss_metric = tf.keras.metrics.Mean()

    def get_g_loss(self, d_fake):
        loss = self.wl(tf.ones_like(d_fake), d_fake)
        return loss
    
    def get_d_loss(self, target, output):
        return self.wl(target, output) + tf.add_n(self.d.losses)
    
    def get_c_loss(self, one_hot, c_pred):
        return self.cfce(one_hot, c_pred)
    
    @tf.function
    def train_wgan(self, batch):
        image, condition = batch["data"], batch["condition_label"]
        for _ in range(self.d_iteration):
            with tf.GradientTape() as d_tape_true:
                noise = tf.random.normal([setting.batch_size, setting.feature_size])
                noise = tf.concat([noise, tf.cast(condition, tf.float32)], 1)
                
                d_true = self.d(noise, training=True)

                d_loss_true = self.get_d_loss(tf.ones_like(d_true), d_true)

            d_gradient = d_tape_true.gradient(d_loss_true, self.d.trainable_variables)
            self.d_opt.apply_gradients(zip(d_gradient, self.d.trainable_variables))

            with tf.GradientTape() as d_tape_fake:
                features = self.g(image)
                features = tf.concat([features, tf.cast(condition, tf.float32)], 1)
                
                d_fake = self.d(features, training=True)

                d_loss_fake = self.get_d_loss(tf.zeros_like(d_fake), d_fake)

            d_gradient = d_tape_fake.gradient(d_loss_fake, self.d.trainable_variables)
            self.d_opt.apply_gradients(zip(d_gradient, self.d.trainable_variables))

            self.d_train_acc_metric.update_state(tf.ones_like(d_true), d_true)
            self.d_train_acc_metric.update_state(tf.zeros_like(d_fake), d_fake)

            self.d_train_loss_metric.update_state(d_loss_true)
            self.d_train_loss_metric.update_state(d_loss_fake)

        for _ in range(self.g_iteration):
            with tf.GradientTape() as g_tape:
                features = self.g(image, training=True)
                features = tf.concat([features, tf.cast(condition, tf.float32)], 1)
                d_fake = self.d(features)

                g_loss = self.get_g_loss(d_fake)

            g_gradient = g_tape.gradient(g_loss, self.g.trainable_variables)
            self.g_opt.apply_gradients(zip(g_gradient, self.g.trainable_variables))

            self.g_train_acc_metric.update_state(tf.ones_like(d_fake), d_fake)

            self.g_train_loss_metric.update_state(g_loss)

    @tf.function
    def train_classifier(self, batch):
        image, one_hot = batch["data"], batch["one_hot_coding_label"]
        with tf.GradientTape() as c_tape:
            features = self.g(image)
            c_pred = self.c(features, training=True)

            c_loss = self.get_c_loss(one_hot, c_pred)
        
        c_gradient = c_tape.gradient(c_loss, self.c.trainable_variables)
        self.c_opt.apply_gradients(zip(c_gradient, self.c.trainable_variables))

        self.c_train_acc_metric.update_state(one_hot, c_pred)

        self.c_train_loss_metric.update_state(c_loss)

    @tf.function
    def test_wgan(self, batch):
        image, condition = batch["data"], batch["condition_label"]

        noise = tf.random.normal([setting.batch_size, setting.feature_size])
        noise = tf.concat([noise, tf.cast(condition, tf.float32)], 1)
        features = self.g(image)
        features = tf.concat([features, tf.cast(condition, tf.float32)], 1)

        d_true = self.d(noise)
        d_fake = self.d(features)

        d_loss_true = self.get_d_loss(tf.ones_like(d_true), d_true)
        d_loss_fake = self.get_d_loss(tf.zeros_like(d_fake), d_fake)
        g_loss = self.get_g_loss(d_fake)

        self.d_test_acc_metric.update_state(tf.ones_like(d_true), d_true)
        self.d_test_acc_metric.update_state(tf.zeros_like(d_fake), d_fake)

        self.d_test_loss_metric.update_state(d_loss_true)
        self.d_test_loss_metric.update_state(d_loss_fake)

        self.g_test_acc_metric.update_state(tf.ones_like(d_fake), d_fake)

        self.g_test_loss_metric.update_state(g_loss)

    @tf.function
    def test_classifier(self, batch):
        image, one_hot = batch["data"], batch["one_hot_coding_label"]

        features = self.g(image)
        c_pred = self.c(features)

        c_loss = self.get_c_loss(one_hot, c_pred)

        self.c_test_acc_metric.update_state(one_hot, c_pred)

        self.c_test_loss_metric.update_state(c_loss)

    def train(self, epoch, train_dataset, validation_dataset):
        train_dataset = train_dataset.shuffle(train_dataset.cardinality(), reshuffle_each_iteration=True)
        validation_dataset = validation_dataset.shuffle(validation_dataset.cardinality(), reshuffle_each_iteration=True)

        for epoch_num in range(epoch):
            start = time.time()
            
            self.g_train_acc_metric.reset_state()
            self.d_train_acc_metric.reset_state()
            self.c_train_acc_metric.reset_state()

            self.g_train_loss_metric.reset_state()
            self.d_train_loss_metric.reset_state()
            self.c_train_loss_metric.reset_state()

            self.g_test_acc_metric.reset_state()
            self.d_test_acc_metric.reset_state()
            self.c_test_acc_metric.reset_state()

            self.g_test_loss_metric.reset_state()
            self.d_test_loss_metric.reset_state()
            self.c_test_loss_metric.reset_state()

            for batch in train_dataset.batch(setting.batch_size, drop_remainder=True):
                self.train_wgan(batch)
                self.train_classifier(batch)
            for batch in validation_dataset.batch(setting.batch_size, drop_remainder=True):
                self.test_wgan(batch)
                self.test_classifier(batch)

            self.g.save(setting.wgan_generator_path)
            self.d.save(setting.wgan_discriminator_path)
            self.c.save(setting.classifier_path)

            cprint('Time for epoch {} is {} sec'.format(epoch_num + 1, time.time()-start), 'red')

            print("Train Generator Loss: " + str(self.g_train_loss_metric.result().numpy()))
            print("Test Generator Loss: " + str(self.g_test_loss_metric.result().numpy()))

            print("Train Generator Accuraccy: " + str(self.g_train_acc_metric.result().numpy()))
            print("Test Generator Accuraccy: " + str(self.g_test_acc_metric.result().numpy()))

            print("Train Discriminator Loss: " + str(self.d_train_loss_metric.result().numpy()))
            print("Test Discriminator Loss: " + str(self.d_test_loss_metric.result().numpy()))

            print("Train Discriminator Accuraccy: " + str(self.d_train_acc_metric.result().numpy()))
            print("Test Discriminator Accuraccy: " + str(self.d_test_acc_metric.result().numpy()))

            print("Train Classifier Loss: " + str(self.c_train_loss_metric.result().numpy()))
            print("Test Classifier Loss: " + str(self.c_test_loss_metric.result().numpy()))

            print("Train Classifier Accuraccy: " + str(self.c_train_acc_metric.result().numpy()))
            print("Test Classifier Accuraccy: " + str(self.c_test_acc_metric.result().numpy()))

            