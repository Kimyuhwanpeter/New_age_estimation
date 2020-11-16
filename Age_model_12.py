# -*- coding: utf-8 -*-
import tensorflow as tf

class layers():
    Conv2D = tf.keras.layers.Conv2D
    BatchNorm = tf.keras.layers.BatchNormalization
    ReLU = tf.keras.layers.ReLU

def res_block(x, filters, weight_decay, i=0):

    h = layers.Conv2D(filters=filters,
                               kernel_size=1,
                               strides=2 if i == 0 else 1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    h = layers.BatchNorm()(h)
    h = layers.ReLU()(h)
    h = layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = layers.BatchNorm()(h)
    h = layers.ReLU()(h)
    h = layers.Conv2D(filters=filters*4,
                               kernel_size=1,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = layers.BatchNorm()(h)

    if i == 0:
        h = layers.ReLU()(h)
    else:
        h = layers.ReLU()(h + x)    
    
    return h


def ensemble_model(input_shape=(64, 64, 1), num_classes=60, decay=0.00005):

    h = inputs = tf.keras.Input(input_shape)
    labels = tf.keras.layers.Flatten()(h)    # 이렇게 하려면 [0, 1] 로 normalzation 시켜야한다
    # 앙살블방식으로 모델을 짜자!!! --> 유사 앙상블말고 진짜 앙상블처럼! 부스팅 알고리즘같이!!

    h = tf.pad(h, [[0,0], [3,3], [3,3], [0,0]], mode="REFLECT")
    h = layers.Conv2D(filters=64,
                      kernel_size=7,
                      strides=1,
                      padding="valid",
                      use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(decay))(h)
    h = layers.BatchNorm()(h)
    h = layers.ReLU()(h)    # 64 x 64 x 64

    h = layers.Conv2D(filters=128,
                      kernel_size=3,
                      strides=2,
                      padding="same",
                      use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(decay))(h)
    h = layers.BatchNorm()(h)
    h = layers.ReLU()(h)    # 32 x 32 x 128

    for i in range(2):
        h = res_block(h, 64, decay, i)
    for i in range(3):
        h = res_block(h, 128, decay, i)
    for i in range(4):
        h = res_block(h, 256, decay, i)
    for i in range(1):
        h = res_block(h, 356, decay, 0)
    h = layers.Conv2D(filters=4096,
                      kernel_size=1,
                      strides=1,
                      padding="same")(h)

    h = tf.keras.layers.GlobalAveragePooling2D()(h) # [batch, 4096]
    h = -labels * tf.math.log(tf.nn.softmax(h) + 0.0000001) # 여기서의 labels은 입력 이미지의 픽셀들이다.

    h = tf.keras.layers.Reshape([64, 64, 1])(h)

    h = tf.pad(h, [[0,0], [3,3], [3,3], [0,0]], mode="REFLECT")
    h = layers.Conv2D(filters=64,
                      kernel_size=7,
                      strides=1,
                      padding="valid",
                      use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(decay))(h)
    h = layers.BatchNorm()(h)
    h = layers.ReLU()(h)    # 64 x 64 x 64

    h = layers.Conv2D(filters=128,
                      kernel_size=3,
                      strides=2,
                      padding="same",
                      use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(decay))(h)
    h = layers.BatchNorm()(h)
    h = layers.ReLU()(h)    # 32 x 32 x 128

    for i in range(2):
        h = res_block(h, 64, decay, i)
    for i in range(3):
        h = res_block(h, 128, decay, i)
    for i in range(4):
        h = res_block(h, 256, decay, i)
    for i in range(1):
        h = res_block(h, 356, decay, 0)
    h = layers.Conv2D(filters=4096,
                      kernel_size=1,
                      strides=1,
                      padding="same")(h)
    h = tf.keras.layers.GlobalAveragePooling2D()(h) # [batch, 4096]
    h = -labels * tf.math.log(tf.nn.softmax(h) + 0.0000001) # 여기서의 labels은 입력 이미지의 픽셀들이다.

    h = tf.keras.layers.Reshape([64, 64, 1])(h)

    h = tf.pad(h, [[0,0], [3,3], [3,3], [0,0]], mode="REFLECT")
    h = layers.Conv2D(filters=64,
                      kernel_size=7,
                      strides=1,
                      padding="valid",
                      use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(decay))(h)
    h = layers.BatchNorm()(h)
    h = layers.ReLU()(h)    # 64 x 64 x 64

    h = layers.Conv2D(filters=128,
                      kernel_size=3,
                      strides=2,
                      padding="same",
                      use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(decay))(h)
    h = layers.BatchNorm()(h)
    h = layers.ReLU()(h)    # 32 x 32 x 128

    for i in range(2):
        h = res_block(h, 64, decay, i)
    for i in range(3):
        h = res_block(h, 128, decay, i)
    for i in range(4):
        h = res_block(h, 256, decay, i)
    for i in range(1):
        h = res_block(h, 356, decay, 0)
    h = layers.Conv2D(filters=4096,
                      kernel_size=1,
                      strides=1,
                      padding="same")(h)
    h = tf.keras.layers.GlobalAveragePooling2D()(h) # [batch, 4096]
    h = tf.keras.layers.Dense(num_classes)(h)       # 입력이미지가 워낙 작아서 이게 될지모르겠다...

    return tf.keras.Model(inputs=inputs, outputs=h)