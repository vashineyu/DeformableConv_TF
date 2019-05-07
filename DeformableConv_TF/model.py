# model.py

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from .layers.deformable_layers import DeformableConvLayer
from .backbone import *

graph_mapping = {
    "R-50-v1":ResNet50,
    "R-101-v1":ResNet101,
    "R-152-v1":ResNet152,
    "R-50-v2":ResNet50V2,
    "R-101-v2":ResNet101V2,
    "R-152-v2":ResNet152V2,
    "R-50-xt":ResNeXt50,
    "R-101-xt":ResNeXt101}

def build_model(input_shape, output_num, use_deformable=False, num_deform_group=0):
    num_deform_group = None if num_deform_group == 0 else num_deform_group
    input_layer = layers.Input(shape=input_shape, dtype=tf.float32)
    
    x = layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding="same", backbone=None)(input_layer)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    if use_deformable:
        x = DeformableConvLayer(filters=64, 
                                kernel_size=3, 
                                strides=(1,1), 
                                padding="same", 
                                num_deformable_group=num_deform_group)(x)
    else:
        x = layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    if use_deformable:
        x = DeformableConvLayer(filters=64, 
                                kernel_size=3, 
                                strides=(1,1), 
                                padding="same", 
                                num_deformable_group=num_deform_group)(x)
    else:
        x = layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding="same")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    out = layers.Dense(units=output_num, activation="softmax")(x)
    return Model(inputs=[input_layer], outputs=[out])
    
def build_resnet_model(input_shape, output_num, use_deformable=False, num_deform_group=0, backbone="R-50-v1"):
    num_deform_group = None if num_deform_group == 0 else num_deform_group
    model_fn = graph_mapping[backbone]
    pretrain_modules = model_fn(include_top=False, 
                                input_shape=input_shape, 
                                norm_use="bn", 
                                weights=None, 
                                use_deformable=use_deformable)
    gap = tf.keras.layers.GlobalAveragePooling2D()(pretrain_modules.output)
    logits = tf.keras.layers.Dense(units=output_num, name="logits")(gap)
    output = tf.keras.layers.Activation("softmax", name="output")(logits)
    
    return tf.keras.Model(inputs=pretrain_modules.input, outputs=output)
    
def preproc_fn(x):
    return x / 255.