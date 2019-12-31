import os, sys, cv2
import shutil
import numpy as np

try:
    video_in, out_dir = str(sys.argv[1]), str(sys.argv[2])
except IndexError:
    print('Format: python {} [video_in] [out_dir]'.format(sys.argv[0]))
    sys.exit(1)

# out_dir = out_dir.encode('unicode_escape')
vid = cv2.VideoCapture(video_in)

shutil.rmtree('{}'.format(out_dir), ignore_errors=True)
os.mkdir(out_dir)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cnt = 0
while(True):
    ret, img = vid.read()
    if ret:
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = sorted(face_cascade.detectMultiScale(gray_frame, 1.3, 5), key=lambda x: -x[0])
        x, y, w, h = faces[0]
        
        s = os.path.join(out_dir,'{:03d}.jpg'.format(cnt))
        # print(s)
        # cv2.imencode('.jpg', gray_frame[y:y+h,x:x+w])[1].tofile(s)
        if not cv2.imwrite(s, gray_frame[y:y+h,x:x+w]):
            raise Exception("Could not write image")
        cnt += 1
    else:
        break

print(cnt)
vid.release()

start, end = cnt//5, cnt//5*4
for i in range(start): os.remove(os.path.join(out_dir,'{:03d}.jpg'.format(i)))
for i in range(end, cnt): os.remove(os.path.join(out_dir,'{:03d}.jpg'.format(i)))
for i in range(start, end): os.rename(os.path.join(out_dir,'{:03d}.jpg'.format(i)), os.path.join(out_dir,'{:03d}.jpg'.format(i-start)))
