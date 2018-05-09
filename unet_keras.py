"""U-Net model.
See https://arxiv.org/pdf/1505.04597
Code was modified from
https://github.com/jocicmarko/ultrasound-nerve-segmentation
"""


# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import os
# import random
from keras.models import Model
from keras.layers import Input, Conv2D, concatenate, Conv2DTranspose, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Dense, BatchNormalization, Activation
from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
# from sklearn.metrics import jaccard_similarity_score
# from collections import defaultdict
from tensorflow.python.ops.clip_ops import clip_by_value

smooth = 1e-12


def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def make_conv_block(nb_filters, input_tensor, block):
    def make_stage(input_tensor, stage):
        name = 'conv_{}_{}'.format(block, stage)
        x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', name=name)(input_tensor)
        name = 'batch_norm_{}_{}'.format(block, stage)
        x = BatchNormalization(name=name)(x)
        x = Activation('relu')(x)
        return x

    x = make_stage(input_tensor, 1)
    x = make_stage(x, 2)
    return x

def get_unet(TailleTuile, NbreClasses):
    
    inputs = Input((3, TailleTuile, TailleTuile))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    batch1 = BatchNormalization(axis=1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(batch1)
 
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    batch2 = BatchNormalization(axis=1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(batch2)
 
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    batch3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(batch3)
 
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    batch4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(batch4)
 
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    batch5 = BatchNormalization(axis=1)(conv5)
    
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(batch5), batch4], axis=1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    batch6 = BatchNormalization(axis=1)(conv6)
 
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(batch6), batch3], axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    batch7 = BatchNormalization(axis=1)(conv7)
 
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(batch7), batch2], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    batch8 = BatchNormalization(axis=1)(conv8)
 
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(batch8), batch1], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    batch9 = BatchNormalization(axis=1)(conv9) 
    
    conv10 = Conv2D(NbreClasses, (1, 1), activation='softmax', kernel_initializer='he_normal')(batch9)
    

    model = Model(inputs=inputs, outputs=conv10)
    # model.compile(optimizer=Adam(lr=1e-3, decay=0.2), loss='categorical_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'categorical_accuracy'])
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # model.summary()
    return model

def get_unet_Conv_Block(TailleTuile, NbreClasses):
    
    inputs = Input((3, TailleTuile, TailleTuile))
    conv1 = make_conv_block(32, inputs, 1)
    # conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # batch1 = BatchNormalization(axis=1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
 
    conv2 = make_conv_block(64, pool1, 2)
    # conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    # conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    # batch2 = BatchNormalization(axis=1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
 
    conv3 = make_conv_block(128, pool2, 3)
    # conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    # conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # batch3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
 
    conv4 = make_conv_block(256, pool3, 4)
    # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # batch4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
 
    conv5 = make_conv_block(512, pool4, 5)
    # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    #batch5 = BatchNormalization(axis=1)(conv5)
    
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=1)
    # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    # batch6 = BatchNormalization(axis=1)(conv6)
    conv6 = make_conv_block(256, up6, 6)
 
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=1)
    # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    # batch7 = BatchNormalization(axis=1)(conv7)
    conv7 = make_conv_block(128, up7, 7)
 
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=1)
    # conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    # conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    # batch8 = BatchNormalization(axis=1)(conv8)
    conv8 = make_conv_block(64, up8, 8)
 
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=1)
    # conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    # conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # batch9 = BatchNormalization(axis=1)(conv9)
    conv9 = make_conv_block(32, up9, 9) 
    
    # conv10 = Conv2D(NbreClasses, (1, 1), activation='softmax', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(NbreClasses, (1, 1))(conv9)
    
    output = Reshape((TailleTuile * TailleTuile, NbreClasses))(conv10)
    output = Activation('softmax')(output)
    output = Reshape((NbreClasses, TailleTuile, TailleTuile))(output)
    
    
    model = Model(inputs=inputs, outputs=output)
    # model.compile(optimizer=Adam(lr=1e-3, decay=0.2), loss='categorical_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'categorical_accuracy'])
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # model.summary()
    return model
