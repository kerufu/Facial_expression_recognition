import tensorflow as tf

from model import custom_conv2d, custom_dense
import setting

model = tf.keras.Sequential([
    custom_conv2d(64, 3),
    custom_conv2d(128, 5),
    custom_conv2d(512, 3),
    custom_conv2d(512, 3),
    tf.keras.layers.Flatten(),
    custom_dense(512),
    tf.keras.layers.Dense(setting.num_classes)
])



