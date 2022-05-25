# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:24:01 2022

@author: John
"""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, MaxPooling3D, Dropout, Conv3D, Input, GlobalAveragePooling3D, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import Model

def conv3D (x,filters,bn,maxp=0,rdmax=0,drop=True,DepthPool=False):
    if bn==1:
        x = BatchNormalization()(x)    
    x  = Conv3D(filters, (3,3,3), padding='valid', activation='relu')(x)
    
    
    if maxp ==1 and DepthPool==False:
        x = MaxPooling3D((1,2, 2),padding='valid')(x)
    if  DepthPool==True:   
        x = MaxPooling3D((2,2, 2),padding='valid')(x)
        

        

        
    if rdmax == 1:
        x = MaxPooling3D((2,2, 2),padding='valid')(x)
    if drop==True:
        x = Dropout(0.4)(x)
    return x


def make_3d_main (OPTIONS_MODE,OPTIONS_PREPROCESSING,OPTIONS_TRAINING):
    classes = OPTIONS_TRAINING['classes']
    
    str_ac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    str_ac = conv3D (str_ac_in,filters=16,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    str_ac = conv3D (str_ac,32,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    str_ac = conv3D (str_ac,64,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    str_ac = conv3D (str_ac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #str_ac = conv3D (str_ac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #str_ac = conv3D (str_ac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #str_ac = GlobalAveragePooling3D()(str_ac)
    str_ac= tf.keras.layers.Flatten()(str_ac)

    str_nac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    str_nac = conv3D (str_nac_in,filters=16,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    str_nac = conv3D (str_nac,32,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    str_nac = conv3D (str_nac,64,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    str_nac = conv3D (str_nac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #str_nac = conv3D (str_nac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #str_nac = conv3D (str_nac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #str_nac = GlobalAveragePooling3D()(str_nac)      
    str_nac= tf.keras.layers.Flatten()(str_nac)

    res_nac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    res_nac = conv3D (res_nac_in,filters=16,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    res_nac = conv3D (res_nac,32,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    res_nac = conv3D (res_nac,64,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    res_nac = conv3D (res_nac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #res_nac = conv3D (res_nac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #res_nac = conv3D (res_nac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #res_nac = GlobalAveragePooling3D()(res_nac)
    res_nac= tf.keras.layers.Flatten()(res_nac)
    
    res_ac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    res_ac = conv3D (res_ac_in,filters=16,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    res_ac = conv3D (res_ac,32,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    res_ac = conv3D (res_ac,64,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    res_ac = conv3D (res_ac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #res_ac = conv3D (res_ac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #res_ac = conv3D (res_ac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #res_ac = GlobalAveragePooling3D()(res_ac)
    res_ac= tf.keras.layers.Flatten()(res_ac)
        
    n = tf.keras.layers.concatenate([str_ac,str_nac,res_ac,res_nac], axis=-1)
    n = Dense(1500, activation='relu')(n)
    n = Dropout(0.4)(n)
    n = Dense(700, activation='relu')(n)
    n = Dropout(0.4)(n)
    # n = Dense(4096, activation='selu')(c)
    # n = Dropout(0.5)(n)
    #n = Dense(750, activation='elu')(n)
    #n = Dropout(0.5)(n)
    n = Dense(classes, activation='softmax')(n)
    
    
    model = Model(inputs=[str_ac_in,str_nac_in,res_ac_in,res_nac_in], outputs=n)
    
    #opt = SGD(lr=0.01)
    
    opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
    
    opt = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False)
    
    
    model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # model.summary()
    return model




def make_3d_main_70 (OPTIONS_MODE,OPTIONS_PREPROCESSING,OPTIONS_TRAINING):
    ###############################################################
    #  Achieved 70%
    ###############################################################
    classes = OPTIONS_TRAINING['classes']
    
    str_ac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    str_ac = conv3D (str_ac_in,filters=8,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    str_ac = conv3D (str_ac,16,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    str_ac = conv3D (str_ac,32,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    str_ac = conv3D (str_ac,64,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #str_ac = conv3D (str_ac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #str_ac = conv3D (str_ac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #str_ac = GlobalAveragePooling3D()(str_ac)
    str_ac= tf.keras.layers.Flatten()(str_ac)

    str_nac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    str_nac = conv3D (str_nac_in,filters=8,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    str_nac = conv3D (str_nac,16,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    str_nac = conv3D (str_nac,32,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    str_nac = conv3D (str_nac,64,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #str_nac = conv3D (str_nac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #str_nac = conv3D (str_nac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #str_nac = GlobalAveragePooling3D()(str_nac)      
    str_nac= tf.keras.layers.Flatten()(str_nac)

    res_nac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    res_nac = conv3D (res_nac_in,filters=8,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    res_nac = conv3D (res_nac,16,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    res_nac = conv3D (res_nac,32,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    res_nac = conv3D (res_nac,64,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #res_nac = conv3D (res_nac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #res_nac = conv3D (res_nac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #res_nac = GlobalAveragePooling3D()(res_nac)
    res_nac= tf.keras.layers.Flatten()(res_nac)
    
    res_ac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    res_ac = conv3D (res_ac_in,filters=8,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    res_ac = conv3D (res_ac,16,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    res_ac = conv3D (res_ac,32,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    res_ac = conv3D (res_ac,64,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #res_ac = conv3D (res_ac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #res_ac = conv3D (res_ac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #res_ac = GlobalAveragePooling3D()(res_ac)
    res_ac= tf.keras.layers.Flatten()(res_ac)
        
    n = tf.keras.layers.concatenate([str_ac,str_nac,res_ac,res_nac], axis=-1)
    n = Dense(600, activation='relu')(n)
    n = Dropout(0.4)(n)
    n = Dense(300, activation='relu')(n)
    n = Dropout(0.4)(n)
    # n = Dense(4096, activation='selu')(c)
    # n = Dropout(0.5)(n)
    #n = Dense(750, activation='elu')(n)
    #n = Dropout(0.5)(n)
    n = Dense(classes, activation='softmax')(n)
    
    
    model = Model(inputs=[str_ac_in,str_nac_in,res_ac_in,res_nac_in], outputs=n)
    
    #opt = SGD(lr=0.01)
    
    opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
    
    opt = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False)
    
    
    model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # model.summary()
    return model


def make_3d_main_palete (OPTIONS_MODE,OPTIONS_PREPROCESSING,OPTIONS_TRAINING):
    classes = OPTIONS_TRAINING['classes']
    
    str_ac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    str_ac = conv3D (str_ac_in,filters=8,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    str_ac = conv3D (str_ac,16,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #str_ac = conv3D (str_ac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #str_ac = conv3D (str_ac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    str_ac = GlobalAveragePooling3D()(str_ac)

    str_nac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    str_nac = conv3D (str_nac_in,filters=8,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    str_nac = conv3D (str_nac,16,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #str_nac = conv3D (str_nac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #str_nac = conv3D (str_nac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    str_nac = GlobalAveragePooling3D()(str_nac)      
    

    res_nac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    res_nac = conv3D (res_nac_in,filters=8,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    res_nac = conv3D (res_nac,16,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #res_nac = conv3D (res_nac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #res_nac = conv3D (res_nac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    res_nac = GlobalAveragePooling3D()(res_nac)
    
    res_ac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    res_ac = conv3D (res_ac_in,filters=8,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    res_ac = conv3D (res_ac,16,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #res_ac = conv3D (res_ac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #res_ac = conv3D (res_ac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    res_ac = GlobalAveragePooling3D()(res_ac)
    
    n = tf.keras.layers.concatenate([str_ac,str_nac,res_ac,res_nac], axis=-1)
    n = Dense(70, activation='relu')(n)
    n = Dropout(0.3)(n)
    n = Dense(35, activation='relu')(n)
    n = Dropout(0.5)(n)
    # n = Dense(4096, activation='selu')(c)
    # n = Dropout(0.5)(n)
    #n = Dense(750, activation='elu')(n)
    #n = Dropout(0.5)(n)
    n = Dense(classes, activation='softmax')(n)
    
    
    model = Model(inputs=[str_ac_in,str_nac_in,res_ac_in,res_nac_in], outputs=n)
    
    #opt = SGD(lr=0.01)
    
    opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # model.summary()
    return model