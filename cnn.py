from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Activation,GaussianNoise,Flatten,Dropout,BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers
from keras.callbacks import CSVLogger
from keras.optimizers  import Adam
import numpy as np
import random
import sys
import os

try:
    au_name, json_path, in_dir, split_log, model_path = str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5])
except IndexError:
    print('Format: python {} [au_name] [json_path] [in_dir] [split_log] [model_path]'.format(sys.argv[0]))
    print('      : python {} [au_name] [json_path] [in_dir] [split_log] [model_path] [batchSize]'.format(sys.argv[0]))
    sys.exit(1)

if not os.path.exists(json_path):
    print('Json file {} does not exist'.format(json_path))
    sys.exit(1)

au2vids = None
with open(json_path,'r') as reader: au2vids = json.loads(reader.read())
if au_name not in au2vids:
    print('action unit {} does not exist'.format(au_name))
    sys.exit(1)

vids1 = sorted([file.strip('.npy') for file in os.listdir(in_dir)], key = lambda vid: int(vid))
vids = []
# delete empty npys
for vid in vids1:
    features = np.load(in_dir + '{}.npy'.format(vid))
    if features.shape[0] == 0: os.remove(in_dir + '{}.npy'.format(vid))
    else: vids.append(vid)

isPos = set(au2vids[au_name])
trainp_vids, trainn_vids, testp_vids, testn_vids = [], [], [], []
if os.path.exists(split_log):
    isTrain = None
    with open(split_log,'r') as reader: isTrain = json.loads(reader.read())
    for vid in vids:
        if vid in isPos:
            if vid in isTrain: trainp_vids.append(vid)
            else: testp_vids.append(vid)            
        else:
            if vid in isTrain: trainn_vids.append(vid)
            else: testn_vids.append(vid)
else:
    p_vids, n_vids = [], []
    for vid in vids:
        if vid in isPos: p_vids.append(vid)
        else: n_vids.append(vid)
    
    p_num, n_num = len(p_vids), len(n_vids)
    min_num = min(p_num, n_num)
    trainp_num, trainn_num = min_num//8 * 7, min_num//8 * 7
    testp_num, testn_num = p_num-trainp_num, n_num-trainn_num
    
    trainp_vids, testp_vids = p_vids[:trainp_num], p_vids[trainp_num:]
    trainn_vids, testn_vids = n_vids[:trainn_num], n_vids[trainn_num:]
    
    isTrain = dict()
    for vid in trainp_vids+trainn_vids: isTrain[vid] = True
    with open(split_log,'w') as writer: writer.write(json.dumps(isTrain, indent=4))

print('videos          Pos       Neg')
print('Training:     {:>6d}    {:>6d}'.format(len(trainp_vids), len(trainn_vids)))
print(' Testing:     {:>6d}    {:>6d}'.format(len(testp_vids), len(testn_vids)))

feature_dim = np.load(os.path.join(in_dir, '{}.npy'.format(trainp_vids[0]))).shape[1]
trainX, testX, trainY, testY = np.empty((0,feature_dim)), np.empty((0,feature_dim)), np.empty((0,1)), np.empty((0,1))

trainpf_num, trainnf_num, testpf_num, testnf_num = 0, 0, 0, 0
for trainp_vid in trainp_vids:
    features = np.load(os.path.join(in_dir, '{}.npy'.format(trainp_vid)))
    trainX = np.append(trainX, features, axis=0)
    trainY, trainpf_num = np.append(trainY, np.ones((features.shape[0],1))), trainpf_num+features.shape[0]
for trainn_vid in trainn_vids:
    features = np.load(os.path.join(in_dir, '{}.npy'.format(trainn_vid)))
    trainX = np.append(trainX, features, axis=0)
    trainY, trainnf_num = np.append(trainY, np.zeros((features.shape[0],1))), trainnf_num+features.shape[0]
for testp_vid in testp_vids:
    features = np.load(os.path.join(in_dir, '{}.npy'.format(testp_vid)))
    testX = np.append(testX, features, axis=0)
    testY, testpf_num = np.append(testY, np.ones((features.shape[0],1))), testpf_num+features.shape[0]
for testn_vid in testn_vids:
    features = np.load(os.path.join(in_dir, '{}.npy'.format(testn_vid)))
    testX = np.append(testX, features, axis=0)
    testY, testnf_num = np.append(testY, np.zeros((features.shape[0],1))), testnf_num+features.shape[0]



#--------動以下就好--------
datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,\
height_shift_range=0.2,shear_range=0.1,zoom_range=[0.8,1.2],\
fill_mode='constant', horizontal_flip=True)
datagen.fit(trainX)
model = Sequential()
#Layer 0
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
model.add(GaussianNoise(0.1))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(Dropout(0.1))
#Layer 1
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.1))
#Layer 2
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.2))
#Layer 3
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.3))

#Fully connected
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(BatchNormalization(axis = -1,momentum=0.5))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax')) 
print(model.summary())
optim = Adam(lr = 0.001)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy']) 
callbacks = []
csvLogger = CSVLogger("log_cnn.csv", separator=",", append=True)
callbacks.append(csvLogger)
'''
train_history = model.fit(train_x,train_y, batch_size = 640, callbacks=callbacks,\
validation_data = (val_x,val_y), epochs=50, verbose=1, shuffle = True) 
'''

train_history = model.fit_generator(datagen.flow(trainX, trainY, batch_size=128),\
steps_per_epoch=10*len(trainX)/128, epochs=50, verbose=1, shuffle = True,\
validation_data = (testX,testY), callbacks = callbacks)  
model.save('model1.h5')