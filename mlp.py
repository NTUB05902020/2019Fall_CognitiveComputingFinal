from time import time
import pickle
import numpy as np
import sys, os, json
import torch
import torch.nn as nn
from torch.autograd import Variable
from random import random

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

print('videos\t\tPos\tNeg')
print('Training:\t\t{:>6d}\t{:>6d}'.format(len(trainp_vids), len(trainn_vids)))
print('Testing:\t\t{:>6d}\t{:>6d}'.format(len(testp_vids), len(testn_vids)))

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

# mean_x = np.sum(trainX, axis = 0) / np.size(trainX,0)
# var_x = np.var(trainX, axis = 0)
# var_x[var_x == 0] = 1e-10
# trainX = (trainX - mean_x) / np.sqrt(var_x)
# testX = (testX - mean_x) / np.sqrt(var_x)
# np.save(au_name + '_meanX.npy', mean_x)
# np.save(au_name + '_varX.npy', var_x)

print('\nfeatures\t\tPos\tNeg')
print('Training:\t\t{:>6d}\t{:>6d}'.format(trainpf_num, trainnf_num))
print('Testing:\t\t{:>6d}\t{:>6d}'.format(testpf_num, testnf_num))

try:
    batchSize = int(sys.argv[6])
except IndexError:
    batchSize = 128
print('\nBatchSize: {:>4d}\n'.format(batchSize))

# Data Loader (Input Pipeline)
trainset = torch.utils.data.TensorDataset(torch.from_numpy(trainX),torch.from_numpy(trainY))
train_loader = torch.utils.data.DataLoader(dataset=trainset, 
                                           batch_size=batchSize, 
                                           shuffle=True)

testset = torch.utils.data.TensorDataset(torch.from_numpy(testX),torch.from_numpy(testY))
test_loader = torch.utils.data.DataLoader(dataset=testset, 
                                          batch_size=batchSize, 
                                          shuffle=False)


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
net = Net(5900, 256, 2)
net.cuda()

learning_rate = 2e-5
num_epochs = 200
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):  
        # Convert torch tensor to Variable
        features = features.cuda()
        labels = labels.cuda()
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(features.float()).float()
        if random() < 1e-6:
            print("out", features.shape)
            print(labels)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(trainset)//batchSize, loss.item()))
    if epoch % 2 == 0:
        with torch.no_grad():
            correct = 0
            total = 0
            for features, labels in test_loader:
                
                features = features.cuda()
                outputs = net(features.float())
                predicted = torch.max(outputs.data, 1)[1]
                if random() < 0.001:
                    print("test", outputs)
                    print(predicted)
                    print(labels)
                total += labels.size(0)
                correct += (predicted.cpu() == labels.long()).sum()
            print('Test Accuracy: %.2f %%' % (100 * correct / total))
