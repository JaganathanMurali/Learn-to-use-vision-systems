import numpy as np
import cv2 as cv
import glob
frameSize = (640,480)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((24*17,3), np.float32)
objp[:,:2] = np.mgrid[0:24,0:17].T.reshape(-1,2)
objpoints = [] 
imgpoints = [] 
images = glob.glob(r'D:/Experiment 6/New folder/Calib/ima*.png')
for image in images:
 img = cv.imread(image)
 gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 ret, corners = cv.findChessboardCorners(gray, (24,17), None)
 if ret == True:
 objpoints.append(objp)
 corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
 imgpoints.append(corners)
 cv.drawChessboardCorners(img, (24,17), corners2, ret)
 cv.imshow('img', img)
 cv.waitKey(1000)
cv.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, 
None)

np.savez('B.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print('Calibration Completed')
img= cv.imread(r'D:/Experiment 6/New folder/Calib/cali5.png')
cv.imshow('img', img)
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# Undistort
dst = cv.undistort(img, mtx, dist, None, newCameraMatrix)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)
# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.png', dst)
# Reprojection Error
mean_error = 0
for i in range(len(objpoints)):
 imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
 error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
 mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
np.savez('B.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
# Load previously saved data
with np.load('B.npz') as file:
 mtx, dist, rvecs, tvecs = [file[i] for i in ('mtx','dist','rvecs','tvecs')]
def draw(img, corners, imgpts):
 corner = tuple(corners[0].ravel())
 img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 10)
 img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 10)
 img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 10)
 return img
def drawBoxes(img, corners, imgpts):
 imgpts = np.int32(imgpts).reshape(-1,2)
 # draw ground floor in green
 img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
 # draw pillars in blue color
 for i,j in zip(range(4),range(4,8)):
 img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)


# draw top layer in red color
 img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
 return img
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#objp = np.zeros((24*17,3), np.float32)
objp = np.zeros((24*17,3), np.float32)
#objp[:,:2] = np.mgrid[0:24,0:17].T.reshape(-1,2)
objp[:,:2] = np.mgrid[0:24,0:17].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axisBoxes = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
 [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
for image in glob.glob('D:/Experiment 6/New folder/caliResult*.png'):
 img = cv.imread(image)
 gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
 #ret, corners = cv.findChessboardCorners(gray, (24,17),None)
 ret, corners = cv.findChessboardCorners(gray, (24,17), None)
 if ret == True:
 print('Pose Estimation Started')
 corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)
 #print(corners2)
 # Find the rotation and translation vectors.
 ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
 # Project 3D points to image plane
 imgpts, jac = cv.projectPoints(axisBoxes, rvecs, tvecs, mtx, dist)
 img = drawBoxes(img,corners2,imgpts)
 cv.imshow('img',img)
 # k = cv.waitKey(0)
 # if k == ord('s'):
 cv.imwrite('pose.png', img)
cv.waitKey(0)
#cv.destroyAllWindows()
