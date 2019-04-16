#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.models import Sequential
from PIL import Image
import numpy as np
import cv2
import os
from PIL import ImageFile
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
import numpy
import pandas as pd
from sklearn.metrics import accuracy_score
import time
dp_start=time.time()

os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

ImageFile.LOAD_TRUNCATED_IMAGES = True

#Give ur .h5 Path
weights_path='./models/docclassifier_CNN.h5'
classes=['DISO','DLAY','DPID','DSHP','OTHERS','RCRI','RTQM','RZZZ','XXER']



img_width=150
img_height=450

#Creating the Model Architecture
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

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Loading the weights from .h5 File
model.load_weights(weights_path)

#Give your Validation Path 
path='./data_new/validation'


#Loading all files inside all folders to list
fname = []
for root,d_names,f_names in os.walk(path):
    for f in f_names:
        fname.append(os.path.join(root, f))

#Reading Images and Predicting The Output  
list1=[]
df=pd.DataFrame(columns=['Actual','Predicted'])
df1=pd.DataFrame(columns=['Actual','Predicted'])
for img in fname:
    if img.endswith('.jpg'):         
        df1['Actual']=pd.Series(img.split('/')[-2])
        print(df1['Actual'])
        img = cv2.imread(img)       
        #img=img * (1.255)
        img = cv2.resize(img,(150,450))
        im2arr = np.reshape(img,[1,150,450,3])    
        # Predicting the Test set results
        y_pred = model.predict_classes(im2arr)
        #print(y_pred)
        prediction=classes[int(y_pred)]
        df1['Predicted']=pd.Series(prediction)
        print(prediction)
        list1.append(prediction)
        df=df.append(df1,sort=False)
df.to_csv('./prediction.csv',index=False)
accuracy = accuracy_score(df['Actual'], df['Predicted'])
print(list1)
print(accuracy)

dp_end=time.time()
print('\n*****Total Time :',round((dp_end-dp_start)/3600,2),'hours OR mins==>',round((dp_end-dp_start)/60,2),'OR in seconds ==>',(dp_end-dp_start))