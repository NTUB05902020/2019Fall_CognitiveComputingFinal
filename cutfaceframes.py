import os, sys, cv2
import numpy as np

try:
    video_in, out_dir = str(sys.argv[1]), str(sys.argv[2])
except IndexError:
    print('Format: python {} [video_in] [out_dir]'.format(sys.argv[0]))

vid = cv2.VideoCapture(video_in)

os.system('rm -fr {}'.format(out_dir))
os.mkdir(out_dir)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cnt = 0
while(True):
    ret, img = vid.read()
    if ret:
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = sort(face_cascade.detectMultiScale(gray_frame, 1.3, 5), key=lambda x: -x[0])
        x, y, w, h = faces[0]
        cv2.imwrite('{}/{:03d}.jpg'.format(out_dir, cnt), gray_frame[y:y+h,x:x+w])
        cnt += 1
    else:
        break

vid.release()

start, end = cnt//5, cnt//5*4
for i in range(start): os.system('rm -f {}/{:03d}.jpg'.format(out_dir, i))
for i in range(end, cnt): os.system('rm -f {}/{:03d}.jpg'.format(out_dir, i))
for i in range(start, end): os.system('mv {}/{:03d}.jpg {}/{:03d}.jpg'.format(out_dir, i, out_dir, i-start))