import os
import xml.etree.cElementTree as ET
import shutil
import sys
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Activation,GaussianNoise,Flatten,Dropout,BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers
from keras.optimizers  import Adam
from keras.models import load_model
import numpy as np
import random
import cv2
sys.stderr = stderr
try:
    vid_path, img_path, model_path = str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])
except IndexError:
    print('Format: python {} [vid_path] [img_path] [model_path]'.format(sys.argv[0]))
    #print('      : python {} [json_path] [in_dir] [model_path] [batchSize]'.format(sys.argv[0]))
    sys.exit(1)
if not os.path.exists(img_path):
	os.mkdir(img_path)
os.system('python newCutFrames.py ' + vid_path + ' ' + img_path + ' haarcascade_frontalface_default.xml haarcascade_eye.xml')
os.system('python image_align.py ' + img_path)

test_x = []
files = os.listdir(img_path + 'aligned/')
for file in files:
	if file[-4:] != '.jpg':
		continue
	full_file_name = os.path.join(img_path + 'aligned/', file)
	img = cv2.imread(full_file_name)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
	test_x.append(img)

test_x = np.array(test_x)
test_x = test_x.reshape((-1,128,128,1))

model = load_model(model_path)
test_y = model.predict(test_x)  
test_y = np.mean(test_y, axis = 0)
print("Testing result:\nThe probability of Action Unit [1, 4, 6, 12, 15] is:\n",test_y)