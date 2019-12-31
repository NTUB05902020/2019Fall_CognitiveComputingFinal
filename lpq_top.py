import sys
import math
import numpy as np
from scipy import signal
from scipy.spatial import distance_matrix
from skimage import io
from skimage.color import rgb2gray

def lpqtop_(img, V, winSize = np.array([[3.0, 3.0]]), decorr = np.array([0.1, 0.1]),\
	mode = 'nh', planeIdx = 1):
	if np.shape(img)[2] != 1:
		print("input data must have 3 dimensions.")
		return
	if np.shape(winSize) != (1, 2):
		print("windows must have dimension (1*2).")
		return
	if not decorr.any():
		deco = 0
	else:
		deco = 1

	STFTalpha1 = 1./winSize[0][0]
	STFTalpha2 = 1./winSize[0][1]

	img = img.reshape((img.shape[0],img.shape[1])).astype(float)
	r1 = (winSize[0][0] - 1) / 2
	r2 = (winSize[0][1] - 1) / 2
	x1 = np.atleast_2d(np.arange(-r1,r1+1))
	x2 = np.atleast_2d(np.arange(-r2,r2+1))

	## Basic STFT filters
	w01 = (x1*0 + 1)
	w11 = np.exp(np.vectorize(complex)(x1*0, -2 * math.pi * x1 * STFTalpha1))
	w21 = np.conj(w11)
	w02 = (x2*0 + 1)
	w12 = np.exp(np.vectorize(complex)(x2*0, -2 * math.pi * x2 * STFTalpha2))
	w22 = np.conj(w12)

	## filter frequency response
	filterResp = signal.convolve2d(signal.convolve2d(img,w01.T,mode='valid'),\
		w12,mode='valid')
	freqResp = np.zeros((filterResp.shape[0],filterResp.shape[1],8))
	freqResp[:,:,0] = np.real(filterResp)
	freqResp[:,:,1] = np.imag(filterResp)

	filterResp = signal.convolve2d(signal.convolve2d(img,w11.T,mode='valid'),\
		w02,mode='valid')
	freqResp[:,:,2] = np.real(filterResp)
	freqResp[:,:,3] = np.imag(filterResp)

	filterResp = signal.convolve2d(signal.convolve2d(img,w11.T,mode='valid'),\
		w12,mode='valid')
	freqResp[:,:,4] = np.real(filterResp)
	freqResp[:,:,5] = np.imag(filterResp)

	filterResp = signal.convolve2d(signal.convolve2d(img,w11.T,mode='valid'),\
		w22,mode='valid')
	freqResp[:,:,6] = np.real(filterResp)
	freqResp[:,:,7] = np.imag(filterResp)

	freqResp = np.reshape(freqResp, \
		(freqResp.shape[0]*freqResp.shape[1],freqResp.shape[2]))

	if deco == 1:
		freqResp = np.dot(freqResp,V)

	ffreqResp = np.where(freqResp>=0, 1, 0)

	LPQdesc = (ffreqResp[:,0]) + (ffreqResp[:,1]) * 2 \
		+ (ffreqResp[:,2]) * 4 + (ffreqResp[:,3]) * 8 \
		+ (ffreqResp[:,4]) * 16 + (ffreqResp[:,5]) * 32 \
		+ (ffreqResp[:,6]) * 64 + (ffreqResp[:,7]) * 128
	if mode == "nh" or mode == "h":
		LPQdesc, _ = np.histogram(LPQdesc, list(range(256)))
	if mode == "nh":
		LPQdesc = LPQdesc / np.sum(LPQdesc) 
	return LPQdesc

def lpqtopcov_(winSize = np.array([[3.0, 3.0]]), rho_s = 0.1, rho_t = 0.1, planeIdx = 1):
	if winSize.shape != (1,2):
		print("windows must have dimension (1*2).")
		return
	if winSize[0][0] < 3 or winSize[0][1] < 3:
		print("Too Small!!")
		return


	STFTalpha1 = 1./winSize[0][0]
	STFTalpha2 = 1./winSize[0][1]

	r1 = (winSize[0][0] - 1) / 2
	r2 = (winSize[0][1] - 1) / 2
	x1 = np.atleast_2d(np.arange(-r1,r1+1))
	x2 = np.atleast_2d(np.arange(-r2,r2+1))

	## Basic STFT filters
	w01 = (x1*0 + 1)
	w11 = np.exp(np.vectorize(complex)(x1*0, -2 * math.pi * x1 * STFTalpha1))
	w21 = np.conj(w11)
	w02 = (x2*0 + 1)
	w12 = np.exp(np.vectorize(complex)(x2*0, -2 * math.pi * x2 * STFTalpha2))
	w22 = np.conj(w12)

	if planeIdx == 1:
		xp, yp = np.meshgrid(np.linspace(1,winSize[0][1],num = winSize[0][1]),\
			np.linspace(1,winSize[0][0],num = winSize[0][0]))
		pp = np.c_[xp.reshape((xp.shape[0]*xp.shape[1],1)), yp.reshape((yp.shape[0]*yp.shape[1],1))]
		#d_s = np.linalg.norm(pp - pp.T);
		d_s = distance_matrix(pp, pp)
		Cs = rho_s ** d_s

		Ct=1

	elif planeIdx == 2 or planeIdx == 3:
		xp, yp = np.meshgrid(np.ones((1,int(winSize[0][1]))),\
			np.linspace(1,winSize[0][0],num = winSize[0][0]))
		pp = np.c_[xp.reshape((xp.shape[0]*xp.shape[1],1)), yp.reshape((yp.shape[0]*yp.shape[1],1))]
		#d_s = np.linalg.norm(pp - pp.T);
		d_s = distance_matrix(pp, pp)
		Cs = rho_s ** d_s

		xp, yp, zp = np.meshgrid(1, np.ones((1,int(winSize[0][0]))),\
			np.linspace(1,winSize[0][1],num = winSize[0][1]))
		pp = np.c_[xp.reshape((xp.shape[0]*xp.shape[1]*xp.shape[2],1)),\
			yp.reshape((yp.shape[0]*yp.shape[1]*yp.shape[2],1)), \
			zp.reshape((zp.shape[0]*zp.shape[1]*zp.shape[2],1))]
		#d_s = np.linalg.norm(pp - pp.T);
		d_t = distance_matrix(pp, pp)
		Ct = rho_t ** d_t

	C = Cs * Ct

	## Form 2-D filters q1, q2, q3, q4 and corresponding 2-D matrix operator M
	q1 = np.dot(w01.T, w12)
	q2 = np.dot(w11.T, w02)
	q3 = np.dot(w11.T, w12)
	q4 = np.dot(w11.T, w22)
	u1, u2 = np.real(q1), np.imag(q1)
	u3, u4 = np.real(q2), np.imag(q2)
	u5, u6 = np.real(q3), np.imag(q3)
	u7, u8 = np.real(q4), np.imag(q4)
	M = np.r_[u1.reshape((u1.shape[0]*u1.shape[1], 1)).T, u2.reshape((u2.shape[0]*u2.shape[1], 1)).T,\
		u3.reshape((u3.shape[0]*u3.shape[1], 1)).T, u4.reshape((u4.shape[0]*u4.shape[1], 1)).T,\
		u5.reshape((u5.shape[0]*u5.shape[1], 1)).T, u6.reshape((u6.shape[0]*u6.shape[1], 1)).T,\
		u7.reshape((u7.shape[0]*u7.shape[1], 1)).T, u8.reshape((u8.shape[0]*u8.shape[1], 1)).T]
	M = np.fliplr(M)

	## Compute SVD
	D = np.dot(np.dot(M,C),M.T)
	A = np.diag([1.000007, 1.000006,1.000005,1.000004,1.000003,1.000002,1.000001,1.0])
	U, S, V = np.linalg.svd(np.dot(np.dot(A,D),A))

	return U, S, V
	
def LPQ_TOP(Volume, weight_vec = np.array([[1, 1, 1]]), \
	decorr = np.array([0.1, 0.1]), winSize = np.array([5.0, 5.0, 5.0])):
	if np.shape(Volume)[3] != 1:
		print("input data must have 3 dimensions in volume.")
		return
	if np.shape(weight_vec) != (1, 3):
		print("weight vector must have dimension (1*3).")
		return
	if not decorr.any():
		deco = 0
	else:
		deco = 1

	maxsize = np.shape(Volume)[2]
	height = np.shape(Volume)[0]
	width = np.shape(Volume)[1]

	## XY Plane
	if deco:
		C, D, V = lpqtopcov_(np.array([[winSize[0], winSize[1]]]), decorr[0], decorr[1],1)
	else:
		V = 0

	for i in range(maxsize):
		G = Volume[:,:,i].astype(float)
		LPQ_XY = lpqtop_(G, V, np.array([[winSize[0], winSize[1]]]), decorr, 'h', 1).T
		
		if i == 0: 
			Hist_XY = np.zeros((LPQ_XY.shape[0],))
		Hist_XY += LPQ_XY

	## XT Plane
	if deco:
		C, D, V = lpqtopcov_(np.array([[winSize[0], winSize[2]]]), decorr[0], decorr[1],2)
	else:
		V = 0

	for i in range(width):
		G = Volume[:,i,:].astype(float)
		LPQ_XT = lpqtop_(G, V, np.array([[winSize[0], winSize[2]]]), decorr, 'h', 2).T

		if i == 0: 
			Hist_XT = np.zeros((LPQ_XT.shape[0],))
		Hist_XT += LPQ_XT

	## YT Plane
	if deco:
		C, D, V = lpqtopcov_(np.array([[winSize[1], winSize[2]]]), decorr[0], decorr[1],3)
	else:
		V = 0

	for i in range(height):
		G = Volume[i,:,:].astype(float)
		LPQ_YT = lpqtop_(G, V, np.array([[winSize[1], winSize[2]]]), decorr, 'h', 3).T

		if i == 0: 
			Hist_YT = np.zeros((LPQ_YT.shape[0],))
		Hist_YT += LPQ_YT

	## Normalize
	Hist_XY = Hist_XY / np.sum(Hist_XY)
	Hist_XT = Hist_XT / np.sum(Hist_XT)
	Hist_YT = Hist_YT / np.sum(Hist_YT)

	Feature_vector = np.r_[weight_vec[0][0] * Hist_XY, \
	weight_vec[0][1] * Hist_XT, weight_vec[0][2] * Hist_YT]

	return Feature_vector

if __name__ == '__main__':
	image1 = io.imread('4.jpg')
	image2 = io.imread('5.jpg')
	image3 = io.imread('6.jpg')
	image4 = io.imread('7.jpg')
	image5 = io.imread('8.jpg')
	image6 = io.imread('9.jpg')
	image_seq = np.array([rgb2gray(image1), rgb2gray(image2), rgb2gray(image3),\
		rgb2gray(image4), rgb2gray(image5), rgb2gray(image6)])
	image_seq = image_seq.reshape((image_seq.shape[0],image_seq.shape[1],image_seq.shape[2],1))
	print(LPQ_TOP(image_seq).shape)




