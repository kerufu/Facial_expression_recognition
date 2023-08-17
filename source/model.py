import tensorflow as tf

import setting

class WassersteinLoss(tf.keras.losses.Loss):
  def call(self, y_true, y_pred):
    return -tf.math.reduce_mean(y_true*y_pred)

class ClipConstraint(tf.keras.constraints.Constraint):
    
    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -setting.kernal_clip_value, setting.kernal_clip_value)
    
    def get_config(self):
        return {'kernal_clip_value': setting.kernal_clip_value}

class custom_conv2d(tf.keras.layers.Layer):

    def __init__(self, num_channel, kernel_size, maxpooling=False, regularize_kernal=False, clip_kernal=False, dropout=False):
        super(custom_conv2d, self).__init__()

        kernel_regularizer = None
        if regularize_kernal:
            kernel_regularizer = tf.keras.regularizers.L1L2()

        kernel_constraint = None
        if clip_kernal:
            kernel_constraint = ClipConstraint()

        if maxpooling:
            self.model = [
                tf.keras.layers.Conv2D(num_channel, kernel_size, padding='same', kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint),
                tf.keras.layers.MaxPool2D(),
            ]
        else:
            self.model = [
                tf.keras.layers.Conv2D(num_channel, kernel_size, strides=2, padding='same', kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint),
            ]

        self.model += [
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('leaky_relu')
        ]
        if dropout:
            self.model.append(tf.keras.layers.Dropout(setting.dropout_ratio))

    def call(self, x, training):
        for layer in self.model:
            if "dropout" in layer.name or "batch_normalization" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    
class custom_conv2dtp(tf.keras.layers.Layer):

    def __init__(self, num_channel, kernel_size, regularize_kernal=False, clip_kernal=False, dropout=False):
        super(custom_conv2dtp, self).__init__()

        kernel_regularizer = None
        if regularize_kernal:
            kernel_regularizer = tf.keras.regularizers.L1L2()

        kernel_constraint = None
        if clip_kernal:
            kernel_constraint = ClipConstraint()

        self.model = [
            tf.keras.layers.Conv2DTranspose(num_channel, kernel_size, strides=2, padding='same', kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('leaky_relu'),
        ]
        if dropout:
            self.model.append(tf.keras.layers.Dropout(setting.dropout_ratio))

    def call(self, x, training):
        for layer in self.model:
            if "dropout" in layer.name or "batch_normalization" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    
class custom_dense(tf.keras.layers.Layer):

    def __init__(self, output_size, regularize_kernal=False, clip_kernal=False, dropout=False):
        super(custom_dense, self).__init__()

        kernel_regularizer = None
        if regularize_kernal:
            kernel_regularizer = tf.keras.regularizers.L1L2()

        kernel_constraint = None
        if clip_kernal:
            kernel_constraint = ClipConstraint()

        self.model = [
            tf.keras.layers.Dense(output_size, kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('leaky_relu')
        ]
        if dropout:
            self.model.append(tf.keras.layers.Dropout(setting.dropout_ratio))

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
            custom_conv2d(64, 3, maxpooling=True, dropout=True),
            custom_conv2d(128, 5, maxpooling=True, dropout=True),
            custom_conv2d(256, 3, maxpooling=True, dropout=True),
            custom_conv2d(512, 3, maxpooling=True, dropout=True),
            tf.keras.layers.Flatten(),
            custom_dense(setting.feature_size, dropout=True)
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
            custom_dense(setting.image_size*setting.image_size*2, regularize_kernal=True, dropout=True),  
            tf.keras.layers.Reshape((setting.image_size//16, setting.image_size//16, 512)),
            custom_conv2dtp(256, 3, regularize_kernal=True, dropout=True),
            custom_conv2dtp(128, 3, regularize_kernal=True, dropout=True),
            custom_conv2dtp(64, 5, regularize_kernal=True, dropout=True),
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
            custom_dense(128, regularize_kernal=True, dropout=True),
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
            custom_conv2d(32, 3, regularize_kernal=True, dropout=True),
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
            tf.keras.layers.Dense(setting.num_classes)
        ]

    def call(self, x, training=False):
        for layer in self.model:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    
class wgan_generator(tf.keras.Model):
    def __init__(self):
        super(wgan_generator, self).__init__()
        self.model = [
            custom_conv2d(64, 5, dropout=True),
            custom_conv2d(128, 3, dropout=True),
            custom_conv2d(256, 3, dropout=True),
            custom_conv2d(512, 3, dropout=True),
            tf.keras.layers.Flatten(),
            custom_dense(setting.feature_size, dropout=True)
        ]

    def call(self, x, training=False):
        for layer in self.model:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    
class wgan_discriminator(tf.keras.Model):
    def __init__(self):
        super(wgan_discriminator, self).__init__()
        self.model = [
            custom_dense(256, regularize_kernal=True, clip_kernal=True, dropout=True),
            custom_dense(64, regularize_kernal=True, clip_kernal=True, dropout=True),
            tf.keras.layers.Dense(1)
        ]

    def call(self, x, training=False):
        for layer in self.model:
            if "custom" in layer.name:
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    