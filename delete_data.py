import os, sys, cv2
import numpy as np

dest = sys.argv[1]

_, dirs, _ = list(os.walk(dest))[0]
for AU_dir in dirs:
	_, indice, _ = list(os.walk(dest + AU_dir + '/'))[0]
	for index in indice:
		os.system('mv ' + dest + AU_dir + '/' + index + '/aligned/' + 'features_lbp.npy '\
			+ dest + AU_dir + '/' + index + '/features_lbp.npy')
		files = os.listdir(dest + AU_dir + '/' + index)
		for file in files:
			if file == "aligned":
				os.system('rm -r {}'.format(dest + AU_dir + '/' + index + '/' + file))
			elif file[-4:] != '.npy':
				os.system('rm {}'.format(dest + AU_dir + '/' + index + '/' + file))