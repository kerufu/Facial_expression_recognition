import tensorflow as tf

import setting

class encoder(tf.keras.Model):
    def __init__(self):
        super(encoder, self).__init__()
        self.layers = [
            tf.keras.layers.Reshape((setting.image_size, setting.image_size, 1)),
            tf.keras.layers.Conv2D(8, 2, activation='selu', padding='same',
                                   kernel_regularizer=tf.keras.regularizers.L1L2()
                                   ),
            tf.keras.layers.Conv2D(16, 2, activation='selu', padding='same',
                                   kernel_regularizer=tf.keras.regularizers.L1L2()
                                   ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='selu',
                                  kernel_regularizer=tf.keras.regularizers.L1L2()
                                  ),
            tf.keras.layers.Dense(setting.feature_size,
                                  activity_regularizer=tf.keras.regularizers.L1L2(),
                                  kernel_regularizer=tf.keras.regularizers.L1L2()
                                  )
        ]

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class decoder(tf.keras.Model):
    def __init__(self):
        super(decoder, self).__init__()
        self.layers = [
            tf.keras.layers.Dense(setting.image_size*setting.image_size,
                                  kernel_regularizer=tf.keras.regularizers.L1L2()
                                  ),  
            tf.keras.layers.Reshape((setting.image_size//4, setting.image_size//4, 16)),
            tf.keras.layers.Conv2DTranspose(8, 2, strides=2, padding='same',
                                            kernel_regularizer=tf.keras.regularizers.L1L2()
                                            ),
            tf.keras.layers.Conv2DTranspose(1, 2, strides=2, padding='same',
                                            kernel_regularizer=tf.keras.regularizers.L1L2()
                                            )
        ]

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class encoder_discriminator(tf.keras.Model):
    def __init__(self):
        super(encoder_discriminator, self).__init__()
        self.layers = [
            tf.keras.layers.Dense(8, activation='selu'),
            tf.keras.layers.Dense(1)
        ]

    def call(self, input):
        for l in self.layers:
            input = l(input)
        return input
    
class decoder_discriminator(tf.keras.Model):
    def __init__(self):
        super(decoder_discriminator, self).__init__()
        self.model = [
            tf.keras.layers.Conv2D(8, 2, activation='selu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input