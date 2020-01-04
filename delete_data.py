import os, sys, cv2
import numpy as np

dest = sys.argv[1]
_, indice, _ = list(os.walk(dest))[0]
for index in indice:
	os.system('mv ' + dest + index + '/aligned/' + 'features_lpq.npy '\
		+ dest + index + '/features_lpq.npy')
	files = os.listdir(dest + index)
	for file in files:
		if file == "aligned":
			os.system('rm -r {}'.format(dest + index + '/' + file))
		elif file[-4:] != '.npy':
			os.system('rm {}'.format(dest + index + '/' + file))