import setting
from model import encoder, encoder_discriminator


from dataset_worker import dataset_worker

import tensorflow as tf

class model_worker():

    def __init__(self, ed_iteration=1, e_iteration=1):
        self.e_iteration = e_iteration
        self.ed_iteration = ed_iteration

        self.e = encoder()
        self.ed = encoder_discriminator()

        self.e_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.clipnorm)
        
        self.ed_opt = tf.keras.optimizers.Adam(learning_rate=setting.learning_rate, clipnorm=setting.clipnorm)

        self.bfce = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True, label_smoothing=setting.soft_label_ratio)
        
        self.e_train_mean_metric = tf.keras.metrics.Mean()
        self.e_train_std_metric = tf.keras.metrics.Mean()
        self.ed_train_metric = tf.keras.metrics.BinaryAccuracy(threshold=0)

    def get_e_loss(self, ed_fake):
        loss = self.bfce(tf.ones_like(ed_fake), ed_fake)
        return loss
    
    def get_ed_loss(self, target, output):
        return self.bfce(target, output)
    

    def train_step(self, batch):
        image, condition, one_hot = batch["data"], batch["condition_label"], batch["one_hot_coding_label"]
        for _ in range(self.ed_iteration):
            with tf.GradientTape() as ed_tape_true:
                noise = tf.random.normal([setting.batch_size, setting.feature_size])
                
                ed_true = self.ed(noise, training=True)

                ed_loss_true = self.get_ed_loss(tf.ones_like(ed_true), ed_true)

            ed_gradient = ed_tape_true.gradient(ed_loss_true, self.ed.trainable_variables)
            self.ed_opt.apply_gradients(zip(ed_gradient, self.ed.trainable_variables))

            with tf.GradientTape() as ed_tape_fake:
                features = self.e(image)
                
                ed_fake = self.ed(features, training=True)

                ed_loss_fake = self.get_ed_loss(tf.zeros_like(ed_fake), ed_fake)

            ed_gradient = ed_tape_fake.gradient(ed_loss_fake, self.ed.trainable_variables)
            self.ed_opt.apply_gradients(zip(ed_gradient, self.ed.trainable_variables))

            self.ed_train_metric.update_state(tf.ones_like(ed_true), ed_true)
            self.ed_train_metric.update_state(tf.zeros_like(ed_fake), ed_fake)

        for _ in range(self.e_iteration):
            with tf.GradientTape() as e_tape:
                features = self.e(image, training=True)
                ed_fake = self.ed(features)

                e_loss = self.get_e_loss(ed_fake)

            e_gradient = e_tape.gradient(e_loss, self.e.trainable_variables)
            self.e_opt.apply_gradients(zip(e_gradient, self.e.trainable_variables))

            self.e_train_mean_metric.update_state(tf.math.reduce_mean(features))

    def train(self, epoch, train_dataset):
        train_dataset = train_dataset.shuffle(setting.batch_size, reshuffle_each_iteration=True)
        train_dataset = train_dataset.batch(setting.batch_size)

        for epoch_num in range(epoch):
            self.e_train_mean_metric.reset_state()
            self.e_train_std_metric.reset_state()
            self.ed_train_metric.reset_state()

            for batch in train_dataset:
                self.train_step(batch)

            print("Mean: " + str(self.e_train_mean_metric.result().numpy()))
            print("STD: " + str(self.e_train_std_metric.result().numpy()))

            print("D loss: " + str(self.ed_train_metric.result().numpy()))

mw = model_worker()
dw = dataset_worker()
mw.train(50, dw.train_dataset, dw.validation_dataset)


        