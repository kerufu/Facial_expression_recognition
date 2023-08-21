import time

import tensorflow as tf
from termcolor import cprint

import setting
from model import wgan_generator, wgan_discriminator, classifier, WassersteinLoss

class wgan_worker():

    def __init__(self, g_iteration=1, d_iteration=1, c_iteration=1) -> None:
        self.g_iteration = g_iteration
        self.d_iteration = d_iteration
        self.c_iteration = c_iteration

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

        self.wl = WassersteinLoss(label_smoothing=False)
        self.cfce = tf.keras.losses.CategoricalFocalCrossentropy(from_logits=True, label_smoothing=setting.label_smoothing_ratio)

        self.d_train_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)
        self.c_train_acc_metric = tf.keras.metrics.CategoricalAccuracy()

        self.g_train_loss_metric = tf.keras.metrics.Mean()
        self.d_train_loss_metric = tf.keras.metrics.Mean()
        self.c_train_loss_metric = tf.keras.metrics.Mean()

        self.d_test_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)
        self.c_test_acc_metric = tf.keras.metrics.CategoricalAccuracy()

        self.g_test_loss_metric = tf.keras.metrics.Mean()
        self.d_test_loss_metric = tf.keras.metrics.Mean()
        self.c_test_loss_metric = tf.keras.metrics.Mean()

        self.feature_mean = tf.keras.metrics.Mean()
        self.feature_std = tf.keras.metrics.Mean()

    def get_g_loss(self, d_fake, condition, c_pred):
        loss = self.wl(tf.ones_like(d_fake), d_fake) * setting.wgan_discriminator_weight
        loss += self.wl(condition, c_pred)
        return loss
    
    def get_d_loss(self, noise, features, d_true, d_fake):
        loss = self.wl(tf.ones_like(d_true), d_true)
        loss += self.wl(-tf.ones_like(d_fake), d_fake)
        loss += tf.add_n(self.d.losses)
        loss += self.gradient_penalty(noise, features) * setting.gradient_penalty_weight
        return loss
    
    def get_c_loss(self, one_hot, c_pred):
        return self.cfce(one_hot, c_pred)
    
    def gradient_penalty(self, noise, features):
        
        temp_shape = [noise.shape[0]] + [1 for _ in range(len(noise.shape)-1)]
        epsilon = tf.random.uniform(temp_shape, 0.0, 1.0)
        mix = epsilon * noise + (1 - epsilon) * features
        
        with tf.GradientTape() as tape:
            tape.watch(mix)
            d_hat = self.d(mix)
        gradients = tape.gradient(d_hat, mix)
        
        g_norm2 = tf.sqrt(tf.reduce_sum(gradients**2, axis=[dim for dim in range(1, len(noise.shape))]))
        d_regularizer = tf.reduce_mean((g_norm2-1.0)**2)
        return d_regularizer
    
    @tf.function
    def train_wgan_discriminator(self, batch):
        image = batch["data"]
        with tf.GradientTape() as d_tape_true:
            noise = tf.random.normal([setting.batch_size, setting.feature_size])
            features = self.g(image)
            
            d_true = self.d(noise, training=True)
            d_fake = self.d(features, training=True)

            d_loss = self.get_d_loss(noise, features, d_true, d_fake)

        d_gradient = d_tape_true.gradient(d_loss, self.d.trainable_variables)
        self.d_opt.apply_gradients(zip(d_gradient, self.d.trainable_variables))

        self.d_train_acc_metric.update_state(tf.zeros_like(d_fake), d_fake)

        self.d_train_loss_metric.update_state(d_loss)

    @tf.function
    def train_wgan_generator(self, batch):
        image, condition = batch["data"], batch["condition_label"]
        condition = tf.cast(condition, tf.float32)

        with tf.GradientTape() as g_tape:
            features = self.g(image, training=True)
            d_fake = self.d(features)
            c_pred = self.c(features)

            g_loss = self.get_g_loss(d_fake, condition, c_pred)

        g_gradient = g_tape.gradient(g_loss, self.g.trainable_variables)
        self.g_opt.apply_gradients(zip(g_gradient, self.g.trainable_variables))

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
    def test_step(self, batch):
        image, condition, one_hot = batch["data"], batch["condition_label"], batch["one_hot_coding_label"]
        condition = tf.cast(condition, tf.float32)

        noise = tf.random.normal([setting.batch_size, setting.feature_size])
        features = self.g(image)

        d_true = self.d(noise)
        d_fake = self.d(features)
        c_pred = self.c(features)

        d_loss = self.get_d_loss(noise, features, d_true, d_fake)
        g_loss = self.get_g_loss(d_fake, condition, c_pred)
        c_loss = self.get_c_loss(one_hot, c_pred)

        self.d_test_acc_metric.update_state(tf.ones_like(d_true), d_true)
        self.d_test_acc_metric.update_state(tf.zeros_like(d_fake), d_fake)

        self.d_test_loss_metric.update_state(d_loss)

        self.g_test_loss_metric.update_state(g_loss)

        self.feature_mean.update_state(tf.math.reduce_mean(features))
        self.feature_std.update_state(tf.math.reduce_std(features))

        self.c_test_acc_metric.update_state(one_hot, c_pred)

        self.c_test_loss_metric.update_state(c_loss)

    def train(self, epoch, train_dataset, validation_dataset):
        batched_validation_dataset = validation_dataset.batch(setting.batch_size, drop_remainder=True)

        for epoch_num in range(epoch):
            start = time.time()
            
            self.d_train_acc_metric.reset_state()
            self.c_train_acc_metric.reset_state()

            self.g_train_loss_metric.reset_state()
            self.d_train_loss_metric.reset_state()
            self.c_train_loss_metric.reset_state()

            self.d_test_acc_metric.reset_state()
            self.c_test_acc_metric.reset_state()

            self.g_test_loss_metric.reset_state()
            self.d_test_loss_metric.reset_state()
            self.c_test_loss_metric.reset_state()

            self.feature_mean.reset_state()
            self.feature_std.reset_state()

            train_dataset = train_dataset.shuffle(train_dataset.cardinality()//setting.shuffle_buffer_size_divider)
            batched_train_dataset = train_dataset.batch(setting.batch_size, drop_remainder=True)

            for _ in range(self.d_iteration):
                for batch in batched_train_dataset:
                    self.train_wgan_discriminator(batch)
            for _ in range(self.g_iteration):
                for batch in batched_train_dataset:
                    self.train_wgan_generator(batch)
            for _ in range(self.c_iteration):
                for batch in batched_train_dataset:
                    self.train_classifier(batch)
                    
            for batch in batched_validation_dataset:
                self.test_step(batch)
                
            if self.g_iteration:
                self.g.save(setting.wgan_generator_path)
            if self.d_iteration:
                self.d.save(setting.wgan_discriminator_path)
            self.c.save(setting.classifier_path)

            cprint('Time for epoch {} is {} sec'.format(epoch_num + 1, time.time()-start), 'red')

            print("Train Generator Loss: " + str(self.g_train_loss_metric.result().numpy()))
            print("Test Generator Loss: " + str(self.g_test_loss_metric.result().numpy()))

            print("Train Discriminator Loss: " + str(self.d_train_loss_metric.result().numpy()))
            print("Test Discriminator Loss: " + str(self.d_test_loss_metric.result().numpy()))

            print("Train Discriminator Accuraccy: " + str(self.d_train_acc_metric.result().numpy()))
            print("Test Discriminator Accuraccy: " + str(self.d_test_acc_metric.result().numpy()))

            print("Feature Mean: " + str(self.feature_mean.result().numpy()))
            print("Feature STD: " + str(self.feature_std.result().numpy()))

            print("Train Classifier Loss: " + str(self.c_train_loss_metric.result().numpy()))
            print("Test Classifier Loss: " + str(self.c_test_loss_metric.result().numpy()))

            print("Train Classifier Accuraccy: " + str(self.c_train_acc_metric.result().numpy()))
            print("Test Classifier Accuraccy: " + str(self.c_test_acc_metric.result().numpy()))
