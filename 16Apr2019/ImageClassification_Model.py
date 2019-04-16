#!/usr/bin/env python

# -*- coding: utf-8 -*-

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.models import Sequential,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
import numpy as np 
import os
import time
dp_start=time.time()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"


# step 1: load data

img_width = 150
img_height = 450
batch_size = 32
train_data_dir = './data_new/train'
valid_data_dir = './data_new/validation'

datagen = ImageDataGenerator(rescale = 1./255)

train_generator = datagen.flow_from_directory(directory=train_data_dir,
											   target_size=(img_width,img_height),
											   classes=['DISO','DLAY','DPID','DSHP','OTHERS','RCRI','RTQM','RZZZ','XXER'],
											   class_mode='categorical',
											   batch_size=32)


validation_generator = datagen.flow_from_directory(directory=valid_data_dir,
											   target_size=(img_width,img_height),
											   classes=['DISO','DLAY','DPID','DSHP','OTHERS','RCRI','RTQM','RZZZ','XXER'],
											   class_mode='categorical',
											   batch_size=32)


# step-2 : build model

model =Sequential()

model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(9))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['categorical_accuracy'])

print('model complied!!')

print('starting training....')
training = model.fit_generator(generator=train_generator, steps_per_epoch=2048 // 16, epochs=50,validation_data=validation_generator,validation_steps=832//16)

print('training finished!!')

print('saving weights to docclassifier_CNN.h5')

model.save_weights('./models/docclassifier_CNN.h5')

dp_end=time.time()
print('\n*****Total Time :',round((dp_end-dp_start)/3600,2),'hours OR mins==>',round((dp_end-dp_start)/60,2),'OR in seconds ==>',(dp_end-dp_start))