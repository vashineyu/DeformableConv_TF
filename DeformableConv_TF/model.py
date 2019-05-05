# model.py

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from .layers.deformable_layers import DeformableConvLayer

def build_model(input_shape, output_num, use_deformable=False):
    input_layer = layers.Input(shape=input_shape, dtype=tf.float32)
    
    x = layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding="same")(input_layer)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    if use_deformable:
        x = DeformableConvLayer(filters=64, kernel_size=3, strides=(1,1), padding="same")(x)
    else:
        x = layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    if use_deformable:
        x = DeformableConvLayer(filters=64, kernel_size=3, strides=(1,1), padding="same")(x)
    else:
        x = layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    out = layers.Dense(units=output_num, activation="softmax")(x)
    return Model(inputs=[input_layer], outputs=[out])
    

def preproc_fn(x):
    return x / 255.