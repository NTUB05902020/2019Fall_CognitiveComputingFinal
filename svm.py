import numpy as np
import sys, os, json

try:
    au_name, json_path, in_dir, split_log, model_path = str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]), str(sys.argv[5])
except IndexError:
    print('Format: python {} [au_name] [json_path] [in_dir] [split_log] [model_path]'.format(sys.argv[0]))
    print('      : python {} [au_name] [json_path] [in_dir] [split_log] [model_path] [batchNum]'.format(sys.argv[0]))
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

trainpX, trainnX, testX = np.empty((0,feature_dim)), np.empty((0,feature_dim)), np.empty((0,feature_dim))
testpf_num, testnf_num = 0, 0

for trainp_vid in trainp_vids:
    features = np.load(os.path.join(in_dir, '{}.npy'.format(trainp_vid)))
    trainpX = np.append(trainpX, features, axis=0)

for trainn_vid in trainn_vids:
    features = np.load(os.path.join(in_dir, '{}.npy'.format(trainn_vid)))
    trainnX = np.append(trainnX, features, axis=0)

for testp_vid in testp_vids:
    features = np.load(os.path.join(in_dir, '{}.npy'.format(testp_vid)))
    testX = np.append(testX, features, axis=0)
    testpf_num += features.shape[0]
    
for testn_vid in testn_vids:
    features = np.load(os.path.join(in_dir, '{}.npy'.format(testn_vid)))
    testX = np.append(testX, features, axis=0)
    testnf_num += features.shape[0]

testY = np.append(np.ones(testpf_num), np.zeros(testnf_num))
trainpf_num, trainnf_num = trainpX.shape[0], trainnX.shape[0]
trainX, trainY = np.append(trainpX, trainnX, axis=0), np.append(np.ones((trainpf_num,1)), np.zeros((trainnf_num,1)), axis=0)

mean_x, var_x = np.mean(trainX, axis=0), np.var(trainX, axis=0)
var_x[var_x==0] = 1e-10
std_x = np.sqrt(var_x)

trainpX = (trainpX-mean_x) / std_x
trainnX = (trainnX-mean_x) / std_x
testX = (testX-mean_x) / std_x
np.save(au_name + '_meanX.npy', mean_x)
np.save(au_name + '_varX.npy', var_x)

print('\nfeatures          Pos       Neg')
print('Training:       {:>6d}    {:>6d}'.format(trainpf_num, trainnf_num))
print(' Testing:       {:>6d}    {:>6d}'.format(testpf_num, testnf_num))

try:
    batchNum = int(sys.argv[6])
except IndexError:
    batchNum = 4
print('\nBatchNum:  {:>4d}\n'.format(batchNum))

batchSizep, batchSizen = trainpX.shape[0]//batchNum, trainnX.shape[0]//batchNum

from time import time
import pickle
from sklearn.utils import shuffle
from sklearn.svm import SVC

svm_model = None
if os.path.exists(model_path):
    with open(model_path, 'rb') as reader: svm_model = pickle.load(reader)
else:
    svm_model = SVC(kernel='linear', probability=True)

train_errorRate, test_errorRate = [], []
if os.path.exists('train_errorRate{}.npy'.format(au_name)):
    train_errorRate = np.load('train_errorRate{}.npy'.format(au_name)).tolist()
if os.path.exists('test_errorRate{}.npy'.format(au_name)):
    test_errorRate = np.load('test_errorRate{}.npy'.format(au_name)).tolist()

# epoch可調
epoch = 20
for i in range(epoch):
    print('epoch = {}'.format(i), end='    ')
    time_seed = int(time())
    trainpX = shuffle(trainpX, random_state=time_seed)
    trainnX = shuffle(trainnX, random_state=time_seed)
    
    indSp, indEp = 0, batchSizep
    indSn, indEn = 0, batchSizen
    
    for j in range(1,batchNum):
        X = np.append(trainpX[indSp:indEp,:], trainnX[indSn:indEn], axis=0)
        Y = np.append(np.ones(batchSizep), np.zeros(batchSizen), axis=0)

        indSp, indSn = indEp, indEn
        indEp, indEn = indSp+batchSizep, indSn+batchSizen
        svm_model.fit(X, Y)
    
    X = np.append(trainpX[indSp:,:], trainnX[indSn:,:], axis=0)
    Y = np.append(np.ones(trainpX.shape[0]-indSp), np.zeros(trainnX.shape[0]-indSn), axis=0)
    svm_model.fit(X, Y)
    
    trainY_ = np.reshape(svm_model.predict(trainX), (-1,1))
    train_errorRate.append(np.count_nonzero(np.not_equal(trainY, trainY_)) / np.size(trainY) * 100)
    
    testY_ = svm_model.predict(testX)
    test_errorRate.append(np.count_nonzero(np.not_equal(testY,testY_)) / np.size(testY) * 100)
    print('{:.4f}    {:.4f}'.format(train_errorRate[-1], test_errorRate[-1]))

with open(model_path, 'wb') as writer: pickle.dump(svm_model, writer)
np.save('train_errorRate{}.npy'.format(au_name), np.array(train_errorRate))
np.save('test_errorRate{}.npy'.format(au_name), np.array(test_errorRate))
"""
from matplotlib import pyplot as plt
plt.plot(train_errorRate, '.', label='train')
plt.plot(test_errorRate, '-', label='test')
#plt.legend()
plt.savefig('errorRate.png')
"""
