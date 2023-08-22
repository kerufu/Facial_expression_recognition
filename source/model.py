import tensorflow as tf

import setting

class custom_conv2d(tf.keras.layers.Layer):

    def __init__(self, num_channel, kernel_size):
        super(custom_conv2d, self).__init__()
        self.model = [
            tf.keras.layers.Conv2D(num_channel, kernel_size, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('leaky_relu'),
            tf.keras.layers.Dropout(setting.dropout_ratio)
        ]

    def call(self, x, training):
        for layer in self.model:
            if "dropout" in layer.name or "batch_normalization" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    
class custom_conv2dtp(tf.keras.layers.Layer):

    def __init__(self, num_channel, kernel_size):
        super(custom_conv2dtp, self).__init__()
        self.model = [
            tf.keras.layers.Conv2DTranspose(num_channel, kernel_size, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('leaky_relu'),
            tf.keras.layers.Dropout(setting.dropout_ratio)
        ]

    def call(self, x, training):
        for layer in self.model:
            if "dropout" in layer.name or "batch_normalization" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    
class custom_dense(tf.keras.layers.Layer):

    def __init__(self, output_size):
        super(custom_dense, self).__init__()
        self.model = [
            tf.keras.layers.Dense(output_size),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('leaky_relu'),
            tf.keras.layers.Dropout(setting.dropout_ratio)
        ]

    def call(self, x, training):
        for layer in self.model:
            if "dropout" in layer.name or "batch_normalization" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x

class encoder(tf.keras.Model):
    def __init__(self):
        super(encoder, self).__init__()
        self.model = [
            custom_conv2d(64, 3),
            custom_conv2d(128, 5),
            custom_conv2d(256, 3),
            custom_conv2d(512, 3),
            tf.keras.layers.Flatten(),
            custom_dense(256),
            tf.keras.layers.Dense(setting.feature_size),
        ]

    def call(self, x, training=False):
        for layer in self.model:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    
class decoder(tf.keras.Model):
    def __init__(self):
        super(decoder, self).__init__()
        self.model = [
            custom_dense(256),
            custom_dense(setting.image_size*setting.image_size*2),  
            tf.keras.layers.Reshape((setting.image_size//16, setting.image_size//16, 512)),
            custom_conv2dtp(256, 3),
            custom_conv2dtp(128, 3),
            custom_conv2dtp(64, 5),
            tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same')
        ]

    def call(self, x, training=False):
        for layer in self.model:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    
class encoder_discriminator(tf.keras.Model):
    def __init__(self):
        super(encoder_discriminator, self).__init__()
        self.model = [
            custom_dense(128),
            tf.keras.layers.Dense(1)
        ]

    def call(self, x, training=False):
        for layer in self.model:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    
class decoder_discriminator(tf.keras.Model):
    def __init__(self):
        super(decoder_discriminator, self).__init__()
        self.model = [
            custom_conv2d(32, 3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ]

    def call(self, x, training=False):
        for layer in self.model:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    
class classifier(tf.keras.Model):
    def __init__(self):
        super(classifier, self).__init__()
        self.model = [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(setting.num_classes)
        ]

    def call(self, x, training=False):
        for layer in self.model:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    