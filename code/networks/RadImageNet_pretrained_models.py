import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2, ResNet50, InceptionV3, DenseNet121
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,Callback, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers, activations
import os
from time import time
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import argparse
from datetime import datetime


def get_compiled_RadImageNet_model(model_name, image_size, lr, structure = 'unfreezeall', database = "ACDC"):
    if not model_name in ['IRV2', 'ResNet50', 'DenseNet121', 'InceptionV3']:
        raise Exception('Pre-trained network not exists. Please choose IRV2/ResNet50/DenseNet121/InceptionV3 instead')
    else:
        if model_name == 'IRV2':
            if database == 'RadImageNet':
                model_dir ="../RadImageNet_models/RadImageNet-IRV2-notop.h5"
                base_model = InceptionResNetV2(weights=model_dir, input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
            else:
                base_model = InceptionResNetV2(weights='imagenet', input_shape=(image_size, image_size, 3),include_top=False,pooling='avg')
        if model_name == 'ResNet50':
            if database == 'RadImageNet':
                model_dir = "../RadImageNet_models/RadImageNet-ResNet50-notop.h5"
                base_model = ResNet50(weights=model_dir, input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
            else:
                base_model = ResNet50(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
        if model_name == 'DenseNet121':
            if database == 'RadImageNet':
                model_dir = "../RadImageNet_models/RadImageNet-DenseNet121-notop.h5"
                base_model = DenseNet121(weights=model_dir, input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
            else:
                base_model = DenseNet121(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
        if model_name == 'InceptionV3':
            if database == 'RadImageNet':
                model_dir = "../RadImageNet_models/RadImageNet-InceptionV3-notop.h5"
                base_model = InceptionV3(weights=model_dir, input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
            else:
                base_model = InceptionV3(weights='imagenet', input_shape=(image_size, image_size, 3), include_top=False,pooling='avg')
    if structure == 'freezeall':
        for layer in base_model.layers:
            layer.trainable = False
    if structure == 'unfreezeall':
        pass
    if structure == 'unfreezetop10':
        for layer in base_model.layers[:-10]:
            layer.trainable = False
    y = base_model.output
    y = Dropout(0.5)(y)
    predictions = Dense(4, activation='softmax')(y)
    model = Model(inputs=base_model.input, outputs=predictions)
    adam = Adam(learning_rate=lr)
    model.compile(optimizer=adam, loss=BinaryCrossentropy(), metrics=[keras.metrics.AUC(name='auc')])
    return model