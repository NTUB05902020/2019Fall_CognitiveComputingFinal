import numpy as np
import sys, os, json

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
    testp_num, testn_num = p_num//8, n_num//8
    trainp_num, trainn_num = p_num-testp_num, n_num-testn_num
    
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
print('features          Pos       Neg')
print('Training:       {:>6d}    {:>6d}'.format(trainpf_num, trainnf_num))
print(' Testing:       {:>6d}    {:>6d}'.format(testpf_num, testnf_num))

try:
    batchSize = int(sys.argv[6])
except IndexError:
    batchSize = 128
print('\nBatchSize:  {:>4d}\n'.format(batchSize))
divided, batchNum = trainX.shape[0]%batchSize == 0, trainX.shape[0]//batchSize

from time import time
import pickle
from sklearn.utils import shuffle
from sklearn.svm import SVC

svm_model = None
if os.path.exists(model_path):
    with open(model_path, 'rb') as reader: svm_model = pickle.load(reader)
else:
    svm_model = SVC(kernel='rbf', probability=True, verbose=True)

train_errorRate, test_errorRate = [], []
print(trainX.shape, trainY.shape)
# epoch可調
epoch = 10
for i in range(epoch):
    print('epoch = {}'.format(i))
    time_seed = int(time())
    trainX, trainY = shuffle(trainX, trainY, random_state=time_seed)
    indStart, indEnd = 0, batchNum
    
    for j in range(1,batchNum):
        X, Y = trainX[indStart:indEnd], trainY[indStart:indEnd]
        indStart, indEnd = indStart+batchSize, indEnd+batchSize
        svm_model.fit(X, Y)
    if not divided: svm_model.fit(trainX[indStart:], trainY[indStart:])
    
    trainY_ = svm_model.predict(trainX)
    train_errorRate.append(np.count_nonzero(np.not_equal(trainY, trainY_)) / np.size(trainY) * 100)
    
    testY_ = svm_model.predict(testX)
    test_errorRate.append(np.count_nonzero(np.not_equal(testY,testY_)) / np.size(testY) * 100)
with open(model_path, 'wb') as writer: pickle.dump(svm_model, writer)

from matplotlib import pyplot as plt
plt.plot(train_errorRate, '.', label='train')
plt.plot(test_errorRate, '-', label='test')
legend()
plt.savefig('errorRate.png')