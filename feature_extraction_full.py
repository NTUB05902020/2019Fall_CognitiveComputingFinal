import lpq_top
import LPQ
import LBP
import os, sys, cv2
import numpy as np

dest = sys.argv[1]
_, dirs, _ = list(os.walk(dest))[0]
for AU_dir in dirs:
	_, indice, _ = list(os.walk(dest + AU_dir + '/'))[0]
	for index in indice:
		print(dest + AU_dir + '/' + index)
		files = os.listdir(dest + AU_dir + '/' + index + '/aligned/')
		imgs = []
		files.sort()
		if sys.argv[2] == 'lpq_top':
			for file in files:
				if file[-4:] != '.jpg':
					continue
				full_file_name = os.path.join(dest + AU_dir + '/' + index + '/aligned/', file)
				img = cv2.imread(full_file_name)
				img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				imgs.append(img)
			if len(imgs) == 0:
				continue
			imgs = np.array(imgs)
			print(imgs.shape)
			imgs = imgs.reshape((imgs.shape[0],imgs.shape[1],imgs.shape[2],1))
			features_lpq_top = lpq_top.LPQ_TOP(imgs)
			np.save(dest + AU_dir + '/' + index + '/aligned/features.npy', features_lpq_top)
		elif sys.argv[2] == 'lpq':
			lpq_model = LPQ.LPQ(3)
			for file in files:
				if file[-4:] != '.jpg':
					continue
				full_file_name = os.path.join(dest + AU_dir + '/' + index + '/aligned/', file)
				img = cv2.imread(full_file_name)
				img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				feature = lpq_model(img)
				imgs.append(feature)
			features_lpq = np.array(imgs)
			np.save(dest + AU_dir + '/' + index + '/aligned/features_lpq.npy', features_lpq)
		elif sys.argv[2] == 'lbp':
			for file in files:
				if file[-4:] != '.jpg':
					continue
				full_file_name = os.path.join(dest + AU_dir + '/' + index + '/aligned/', file)
				img = cv2.imread(full_file_name)
				img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				feature = LBP.getLBPFeature(img)
				imgs.append(feature)
			features_lbp = np.array(imgs)
			np.save(dest + AU_dir + '/' + index + '/aligned/features_lbp.npy', features_lbp)
	