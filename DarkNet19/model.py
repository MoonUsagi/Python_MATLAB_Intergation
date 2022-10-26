#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    25-Oct-2022 15:01:34

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    input_unnormalized = keras.Input(shape=(256,256,3), name="input_unnormalized")
    input = RescaleZeroOneLayer((256,256,3), name="input_")(input_unnormalized)
    conv1 = layers.Conv2D(32, (3,3), padding="same", name="conv1_")(input)
    batchnorm1 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm1_")(conv1)
    leaky1 = layers.LeakyReLU(alpha=0.100000)(batchnorm1)
    pool1 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(leaky1)
    conv2 = layers.Conv2D(64, (3,3), padding="same", name="conv2_")(pool1)
    batchnorm2 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm2_")(conv2)
    leaky2 = layers.LeakyReLU(alpha=0.100000)(batchnorm2)
    pool2 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(leaky2)
    conv3 = layers.Conv2D(128, (3,3), padding="same", name="conv3_")(pool2)
    batchnorm3 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm3_")(conv3)
    leaky3 = layers.LeakyReLU(alpha=0.100000)(batchnorm3)
    conv4 = layers.Conv2D(64, (1,1), padding="same", name="conv4_")(leaky3)
    batchnorm4 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm4_")(conv4)
    leaky4 = layers.LeakyReLU(alpha=0.100000)(batchnorm4)
    conv5 = layers.Conv2D(128, (3,3), padding="same", name="conv5_")(leaky4)
    batchnorm5 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm5_")(conv5)
    leaky5 = layers.LeakyReLU(alpha=0.100000)(batchnorm5)
    pool3 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(leaky5)
    conv6 = layers.Conv2D(256, (3,3), padding="same", name="conv6_")(pool3)
    batchnorm6 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm6_")(conv6)
    leaky6 = layers.LeakyReLU(alpha=0.100000)(batchnorm6)
    conv7 = layers.Conv2D(128, (1,1), padding="same", name="conv7_")(leaky6)
    batchnorm7 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm7_")(conv7)
    leaky7 = layers.LeakyReLU(alpha=0.100000)(batchnorm7)
    conv8 = layers.Conv2D(256, (3,3), padding="same", name="conv8_")(leaky7)
    batchnorm8 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm8_")(conv8)
    leaky8 = layers.LeakyReLU(alpha=0.100000)(batchnorm8)
    pool4 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(leaky8)
    conv9 = layers.Conv2D(512, (3,3), padding="same", name="conv9_")(pool4)
    batchnorm9 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm9_")(conv9)
    leaky9 = layers.LeakyReLU(alpha=0.100000)(batchnorm9)
    conv10 = layers.Conv2D(256, (1,1), padding="same", name="conv10_")(leaky9)
    batchnorm10 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm10_")(conv10)
    leaky10 = layers.LeakyReLU(alpha=0.100000)(batchnorm10)
    conv11 = layers.Conv2D(512, (3,3), padding="same", name="conv11_")(leaky10)
    batchnorm11 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm11_")(conv11)
    leaky11 = layers.LeakyReLU(alpha=0.100000)(batchnorm11)
    conv12 = layers.Conv2D(256, (1,1), padding="same", name="conv12_")(leaky11)
    batchnorm12 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm12_")(conv12)
    leaky12 = layers.LeakyReLU(alpha=0.100000)(batchnorm12)
    conv13 = layers.Conv2D(512, (3,3), padding="same", name="conv13_")(leaky12)
    batchnorm13 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm13_")(conv13)
    leaky13 = layers.LeakyReLU(alpha=0.100000)(batchnorm13)
    pool5 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(leaky13)
    conv14 = layers.Conv2D(1024, (3,3), padding="same", name="conv14_")(pool5)
    batchnorm14 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm14_")(conv14)
    leaky14 = layers.LeakyReLU(alpha=0.100000)(batchnorm14)
    conv15 = layers.Conv2D(512, (1,1), padding="same", name="conv15_")(leaky14)
    batchnorm15 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm15_")(conv15)
    leaky15 = layers.LeakyReLU(alpha=0.100000)(batchnorm15)
    conv16 = layers.Conv2D(1024, (3,3), padding="same", name="conv16_")(leaky15)
    batchnorm16 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm16_")(conv16)
    leaky16 = layers.LeakyReLU(alpha=0.100000)(batchnorm16)
    conv17 = layers.Conv2D(512, (1,1), padding="same", name="conv17_")(leaky16)
    batchnorm17 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm17_")(conv17)
    leaky17 = layers.LeakyReLU(alpha=0.100000)(batchnorm17)
    conv18 = layers.Conv2D(1024, (3,3), padding="same", name="conv18_")(leaky17)
    batchnorm18 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm18_")(conv18)
    leaky18 = layers.LeakyReLU(alpha=0.100000)(batchnorm18)
    conv19 = layers.Conv2D(1000, (1,1), padding="same", name="conv19_")(leaky18)
    avg1 = layers.GlobalAveragePooling2D(keepdims=True)(conv19)
    softmax = layers.Softmax()(avg1)
    output = layers.Flatten()(softmax)

    model = keras.Model(inputs=[input_unnormalized], outputs=[output])
    return model

## Helper layers:

class RescaleZeroOneLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        super(RescaleZeroOneLayer, self).__init__(name=name)
        self.min = tf.Variable(initial_value=tf.zeros(shape), trainable=False)
        self.max = tf.Variable(initial_value=tf.zeros(shape), trainable=False)

    def call(self, input):
        # Scale the range [min, Max] into [-1, 1]
        return (input - self.min)/(self.max - self.min)

