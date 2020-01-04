import os
import xml.etree.cElementTree as ET
import shutil
import sys
import json
"""def CutFace(video_in, out_dir):	
	vid = cv2.VideoCapture(video_in)
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	cnt = 0
	while(True):
		ret, img = vid.read()
		if ret:
			gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			faces = sorted(face_cascade.detectMultiScale(gray_frame, 1.3, 5), key=lambda x: -x[0])
			x, y, w, h = faces[0]
			cv2.imwrite('{}/{:03d}.jpg'.format(out_dir, cnt), gray_frame[y:y+h,x:x+w])
			cnt += 1
		else:
			break

	vid.release()

	start, end = cnt//5, cnt//5*4
	for i in range(start): os.system('rm -f {}/{:03d}.jpg'.format(out_dir, i))
	for i in range(end, cnt): os.system('rm -f {}/{:03d}.jpg'.format(out_dir, i))
	for i in range(start, end): os.system('mv {}/{:03d}.jpg {}/{:03d}.jpg'.format(out_dir, i, out_dir, i-start))"""

## 抓MMI dataset的所有index以及其對應的xml檔
path = sys.argv[1]

_, dirs, _ = list(os.walk(path))[0]

dict_file = {}
for index in dirs:
	path_index = path + index
	files = os.listdir(path_index)
	for file in files:
		if file[0] == 'S' and "oao" not in file and file[-4:] == ".xml":
			dict_file[index] = file

#print("所有xml檔：", dict_file)

## 將所有data的AU抓出來並放入list

dict_AU = {}
for index, file in dict_file.items():
	tree = ET.ElementTree(file = path + index + "/" + file)
	#print("編號：", index)
	for elem in tree.iter(tag='ActionUnit'):
		#print("AU內容: ", elem.tag, elem.attrib)
		if elem.attrib['Number'] not in dict_AU:
			dict_AU[elem.attrib['Number']] = [index]
		else:
			dict_AU[elem.attrib['Number']].append(index)

print(dict_AU)
with open('dict_AU.json','w') as f:
	json.dump(dict_AU,f)
'''dest = "../MMI_arrange/"
os.mkdir(dest)
for AU_num in dict_AU.keys():
	os.mkdir(dest + AU_num)
	for index in dict_AU[AU_num]:
		os.mkdir(dest + AU_num + '/' + index)
		src_files = os.listdir(path + index)
		for file in src_files:
			full_file_name = os.path.join(path + index, file)
			if os.path.isfile(full_file_name):
				shutil.copy(full_file_name, dest + AU_num + '/' + index)'''

'''import cv2
dest = sys.argv[1]
"""for AU_num in dict_AU.keys():
	for index in dict_AU[AU_num]:
		src_files = os.listdir(dest + AU_num + '/' + index)
		for file in src_files:
			if file[0] == 'S' and file[-4:] == ".avi":
				full_file_name = os.path.join(dest + AU_num + '/' + index, file)
				#os.system('python newCutFrames.py ' + full_file_name + ' ' + dest + AU_num + '/' + index\
				#+ ' haarcascade_frontalface_default.xml haarcascade_eye.xml')
				#os.system('rm '+full_file_name)
				os.system('python image_align.py '+dest + AU_num + '/' + index)"""

_, indice, _ = list(os.walk(dest))[0]
for index in indice:
	print(dest + index)
	files = os.listdir(dest + index)
	for file in files:
		if file[-4:] != '.avi':
			full_file_name = os.path.join(dest + index, file)
			os.system('rm '+full_file_name)
			continue
		full_file_name = os.path.join(dest + index, file)
		os.system('python newCutFrames.py ' + full_file_name + ' ' + dest + index\
			+ ' haarcascade_frontalface_default.xml haarcascade_eye.xml')
		os.system('rm '+full_file_name)
for index in indice:
	os.system('python image_align.py '+ dest + index + '/')'''






