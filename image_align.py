import numpy as np
import cv2
import sys
import os 

files = os.listdir(sys.argv[1])
files.sort()
cnt = 0
os.mkdir(sys.argv[1] + "aligned")
for file in files:
	if file[-4:] != ".jpg":
		continue
	#print(cnt)
	if cnt == 0:
		im1 = cv2.imread(sys.argv[1] + file)
		cnt += 1
		cv2.imwrite('{}aligned/{:03d}.jpg'.format(sys.argv[1], cnt),im1)
		continue
	else:
	# Read the images to be aligned
		im2 = cv2.imread(sys.argv[1] + file)

		# Convert images to grayscale
		im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
		im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

		# Find size of image1
		sz = im1.shape

		# Define the motion model
		warp_mode = cv2.MOTION_TRANSLATION

		# Define 2x3 or 3x3 matrices and initialize the matrix to identity
		if warp_mode == cv2.MOTION_HOMOGRAPHY :
			warp_matrix = np.eye(3, 3, dtype=np.float32)
		else :
			warp_matrix = np.eye(2, 3, dtype=np.float32)

		# Specify the number of iterations.
		number_of_iterations = 5000;

		# Specify the threshold of the increment
		# in the correlation coefficient between two iterations
		termination_eps = 1e-10;

		# Define termination criteria
		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

		# Run the ECC algorithm. The results are stored in warp_matrix.
		(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

		if warp_mode == cv2.MOTION_HOMOGRAPHY :
		# Use warpPerspective for Homography 
			im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
		else :
		# Use warpAffine for Translation, Euclidean and Affine
			im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
		cv2.imwrite('{}aligned/{:03d}.jpg'.format(sys.argv[1], cnt),im2_aligned)
		im1 = im2_aligned
		cnt += 1
