import os, sys, cv2
import numpy as np

def isOverlap(reg0, reg1):
    (x0,y0,w0,h0), (x1,y1,w1,h1) = reg0, reg1
    if x0<=x1 and x1<=x0+w0  or  x1<=x0 and x0<=x1+w1:
        if y0<=y1 and y1<=y0+h0  or  y1<=y0 and y0<=y1+h1: return True
    else: return False

def sortAndClearOverlap(faces):
    ret = sorted(list(faces), key = lambda face: -face[2]*face[3])
    while len(ret) > 1:
        if isOverlap(ret[0], ret[1]): del ret[1]
        else: return ret[:2]
    return ret

def recursiveRemoveOverlap(lis):
    if len(lis) == 1: return lis
    elif isOverlap(lis[0], lis[-1]): return recursiveRemoveOverlap(lis[:-1])
    else: return recursiveRemoveOverlap(lis[:-1]) + lis[-1:]
    
def removeGlasses(eyes):
    if len(eyes) == 1: return eyes
    eyes = recursiveRemoveOverlap(eyes)
    if len(eyes) == 1: return eyes
    else: return eyes[:1] + recursiveRemoveOverlap(eyes[1:])

def detectOrientation(img):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grays = [gray_frame]
    for i in range(3): grays.append(np.rot90(grays[-1]))
    
    facess = []
    for i,gray in enumerate(grays):
        faces = face_cascade.detectMultiScale(gray)
        
        if len(faces) == 1: facess.append(faces)
        elif len(faces) > 1:
            faces = sortAndClearOverlap(faces)
            if len(faces) == 1: facess.append(faces)
            elif i != 0: facess.append([])
            else: facess.append(faces)
        else: facess.append([])
        
        
        outputImg = np.copy(img)
        if i == 0:
            for x,y,w,h in facess[-1]: cv2.rectangle(outputImg, (x,y), (x+w,y+h), (0,0,255), 4)
        elif i == 1:
            for x,y,w,h in facess[-1]: cv2.rectangle(outputImg, (outputImg.shape[1]-y-1,x), (outputImg.shape[1]-y-h-1,x+w), (0,0,255), 4)
        elif i == 2:
            for x,y,w,h in facess[-1]: cv2.rectangle(outputImg, (outputImg.shape[1]-x-1,outputImg.shape[0]-y-1), (outputImg.shape[1]-x-w-1,outputImg.shape[0]-y-h-1), (0,0,255), 4)
        else:
            for x,y,w,h in facess[-1]: cv2.rectangle(outputImg, (y,outputImg.shape[0]-x-1), (y+h,outputImg.shape[0]-x-w-1), (0,0,255), 4)
        cv2.imwrite('output{}.jpg'.format(i), outputImg)
        
    
    orient, maxArea = None, -1
    for i,faces in enumerate(facess):
        if len(faces) == 0: continue
        
        area = faces[0][2]*faces[0][3]
        if area > maxArea:
            orient, maxArea = i, area
            #print(i, len(faces), area)
    
    return (-1,None) if orient == None else (orient,facess[orient])


try:
    video_in, out_dir = str(sys.argv[1]), str(sys.argv[2])
except IndexError:
    print('Format: python {} [video_in] [out_dir]'.format(sys.argv[0]))

os.system('rm -fr {}'.format(out_dir))
os.mkdir(out_dir)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# First judge face orientation in video
"""
face_leftright:
                -1:  only one face in video
                 0:  face is at  left side
                 1:  face is at right side
                 
face_orientation:
                 0:  counter clockwise rotate   0 degree
                 1:                            90
                 2:                           180
                 3:                           270
"""
face_leftright, face_orient = None, None
cnt, vid = 0, cv2.VideoCapture(video_in)
while True:
    ret, img = vid.read()
    if ret:
        cnt += 1
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_orient, faces = detectOrientation(img)
        if face_orient == -1:
            print('Cant\t find face in frame!')
            vid.release()
            sys.exit(1)
        elif len(faces) == 1:
            face_leftright = -1
            break
        else:
            faces = sorted(list(faces), key = lambda face: face[0])
            (x0,y0,w0,h0), (x1,y1,w1,h1) = faces[0], faces[1]
            eyes0 = removeGlasses(sorted(eye_cascade.detectMultiScale(gray_frame[y0:y0+h0,x0:x0+w0]), key = lambda e: e[2]+e[3]))
            eyes1 = removeGlasses(sorted(eye_cascade.detectMultiScale(gray_frame[y1:y1+h1,x1:x1+w1]), key = lambda e: e[2]+e[3]))
            if len(eyes0) < 2:
                if len(eyes1) >= 2:
                    face_leftright = 1
                    break
            else:
                if len(eyes1) < 2:
                    face_leftright = 0
                    break
            print('Still can\'t judge left_right in two-face frame')
    else:
        print('Cant\t find orientation before video end!')
        vid.release()
        sys.exit(1)
vid.release()


#print(cnt, face_leftright, face_orient)

# Start cutting frames
cnt, vid = 0, cv2.VideoCapture(video_in)
while True:
    ret, img = vid.read()
    if ret:
        cnt += 1
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i in range(face_orient): gray_frame = cv2.rot90(gray_frame)
        
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        if len(faces) == 0:
            print('Can\'t find face in frame {}'.format(cnt))
            sys.exit(1)
        
        faces, outImg = sortAndClearOverlap(faces), None
        if len(faces)==1 or face_leftright==-1:
            x,y,w,h = faces[0]
            outImg = gray_frame[y:y+h,x:x+w]
        else:
            faces.sort(key = lambda face: face[0])
            x,y,w,h = faces[face_leftright]
            outImg = gray_frame[y:y+h,x:x+w]
        cv2.imwrite('{}/{:03d}.jpg'.format(out_dir, cnt), outImg)
    else:
        break

vid.release()

start, end = cnt//5, cnt//5*4
for i in range(start): os.system('rm -f {}/{:03d}.jpg'.format(out_dir, i))
for i in range(end, cnt): os.system('rm -f {}/{:03d}.jpg'.format(out_dir, i))
for i in range(start, end): os.system('mv {}/{:03d}.jpg {}/{:03d}.jpg'.format(out_dir, i, out_dir, i-start))
