from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Activation,GaussianNoise,Flatten,Dropout,BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers
from keras.callbacks import CSVLogger
from keras.optimizers  import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import json
import random
import sys
import os
import cv2

try:
    au_name, json_path, in_dir, model_path = str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])+ 'Sessions/', str(sys.argv[4])
except IndexError:
    print('Format: python {} [au_name] [json_path] [in_dir] [model_path]'.format(sys.argv[0]))
    print('      : python {} [au_name] [json_path] [in_dir] [model_path] [batchSize]'.format(sys.argv[0]))
    sys.exit(1)

au2vids = None
with open(json_path,'r') as reader: au2vids = json.loads(reader.read())
if au_name not in au2vids:
    print('action unit {} does not exist'.format(au_name))
    sys.exit(1)

trainX, trainY = [], []
isPos = set(au2vids[au_name])
_, indice, _ = list(os.walk(in_dir))[0]
for index in indice:
    in_path = in_dir + index + '/aligned/'
    print(in_path)
    if not os.path.exists(in_path):
        in_path = in_dir + index
    
    files = os.listdir(in_path)
    files.sort()
    for file in files:
        if file[-4:] != '.jpg':
            continue
        full_file_name = os.path.join(in_path, file)
        img = cv2.imread(full_file_name)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        trainX.append(img)
        if index in isPos:
            trainY.append(1)
        else: trainY.append(0)
    #imgs = imgs.reshape((imgs.shape[0],imgs.shape[1],imgs.shape[2],1))

trainX = np.array(trainX).reshape((-1,128,128,1))
trainY = np.array(trainY)
trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size = 0.2)

#--------動以下就好--------
datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,\
height_shift_range=0.2,shear_range=0.1,zoom_range=[0.8,1.2],\
fill_mode='constant', horizontal_flip=True)
datagen.fit(trainX)
model = Sequential()
#Layer 0
model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', input_shape = (128,128,1)))
model.add(GaussianNoise(0.1))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(Dropout(0.1))
#Layer 1
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.1))
#Layer 2
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.2))
#Layer 3
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.2))
#Layer 3
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.3))

#Fully connected
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 
print(model.summary())
optim = Adam(lr = 0.001)
model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy']) 
#print(model.summary())
callbacks = []
csvLogger = CSVLogger("log_cnn.csv", separator=",", append=True)
callbacks.append(csvLogger)
'''
train_history = model.fit(train_x,train_y, batch_size = 640, callbacks=callbacks,\
validation_data = (val_x,val_y), epochs=50, verbose=1, shuffle = True) 
'''

train_history = model.fit_generator(datagen.flow(trainX, trainY, batch_size=64),\
steps_per_epoch=5*len(trainX)/64, epochs=50, verbose=1, shuffle = True,\
validation_data = (testX,testY), callbacks = callbacks)  
model.save(model_path)