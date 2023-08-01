import tensorflow as tf

import setting

class custom_conv2d(tf.keras.layers.Layer):

    def __init__(self, num_channel, kernel_size):
        super(custom_conv2d, self).__init__()
        self.model = [
            tf.keras.layers.Conv2D(num_channel, kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.L1L2()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('selu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.1)
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    
class custom_conv2dtp(tf.keras.layers.Layer):

    def __init__(self, num_channel, kernel_size, strides):
        super(custom_conv2dtp, self).__init__()
        self.model = [
            tf.keras.layers.Conv2DTranspose(num_channel, kernel_size, strides=strides, padding='same', kernel_regularizer=tf.keras.regularizers.L1L2()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('selu'),
            tf.keras.layers.Dropout(0.1)
            ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    
class custom_dense(tf.keras.layers.Layer):

    def __init__(self, output_size, activity_regularizer=None):
        super(custom_dense, self).__init__()
        self.model = [
            tf.keras.layers.Dense(output_size, kernel_regularizer=tf.keras.regularizers.L1L2(), activity_regularizer=activity_regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('selu'),
            tf.keras.layers.Dropout(0.1)
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x

class encoder(tf.keras.Model):
    def __init__(self):
        super(encoder, self).__init__()
        self.model = [
            custom_conv2d(16, 2),
            custom_conv2d(32, 2),
            tf.keras.layers.Flatten(),
            custom_dense(64),
            custom_dense(setting.feature_size, activity_regularizer=tf.keras.regularizers.L1L2())
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    
class decoder(tf.keras.Model):
    def __init__(self):
        super(decoder, self).__init__()
        self.model = [
            custom_dense(setting.image_size*setting.image_size),  
            tf.keras.layers.Reshape((setting.image_size//8, setting.image_size//8, 64)),
            custom_conv2dtp(32, 2, 2),
            custom_conv2dtp(16, 2, 2),
            custom_conv2dtp(1, 2, 2),
        ]

    def call(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    
class encoder_discriminator(tf.keras.Model):
    def __init__(self):
        super(encoder_discriminator, self).__init__()
        self.model = [
            custom_dense(32),
            custom_dense(16),
            tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L1L2()),
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input
    
class decoder_discriminator(tf.keras.Model):
    def __init__(self):
        super(decoder_discriminator, self).__init__()
        self.model = [
            custom_conv2d(8, 2),
            custom_conv2d(16, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L1L2()),
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input
    
class classifier(tf.keras.Model):
    def __init__(self):
        super(classifier, self).__init__()
        self.model = [
            custom_dense(32),
            custom_dense(16),
            tf.keras.layers.Dense(setting.num_classes, kernel_regularizer=tf.keras.regularizers.L1L2()),
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input
    