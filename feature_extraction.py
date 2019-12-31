import lpq_top
import os, sys, cv2
import numpy as np

dest = sys.argv[1]

_, dirs, _ = list(os.walk(dest))[0]
for AU_dir in dirs:
	_, indice, _ = list(os.walk(dest + AU_dir))[0]
	indice.sort()
	for index in indice:
		files = os.listdir(os.path.join(dest, AU_dir, index, 'aligned'))
		files.sort()
		imgs = []
		for file in files:
			if file[-4:] != '.jpg':
				continue
			full_file_name = os.path.join(dest + AU_dir, index, 'aligned', file)
			# print(full_file_name)
			img = cv2.imread(full_file_name)
			img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			imgs.append(img)
		if len(imgs) == 0:
			continue
		imgs = np.array(imgs)
		print(AU_dir, '-', index, imgs.shape)
		imgs = imgs.reshape((imgs.shape[0],imgs.shape[1],imgs.shape[2],1))
		if imgs.shape[0] < 3:
			continue
		if imgs.shape[0] < 5:
			winsize = imgs.shape[0]
			features_lpq_top = lpq_top.LPQ_TOP(imgs, winSize = np.array([winsize,winsize,winsize]))
		else:
			features_lpq_top = lpq_top.LPQ_TOP(imgs)
			os.path.join(dest + AU_dir, index, 'aligned', 'features_LPQTOP.npy')
		np.save(os.path.join(dest + AU_dir, index, 'aligned', 'features_LPQTOP.npy'), features_lpq_top)
