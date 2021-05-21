import os

import cv2 as cv
import matplotlib.pyplot as plt
import CalibUtils as cut

'''
 checkerboard Dimensions
'''

cbrow = 5
cbcol = 7
inputPath1 = '../../camera_calibration_videos/checkerboard_000.h264'
inputPath2 = '../../camera_calibration_videos/checkerboard_019.h264'
outputPath = "../../camera_calibration_videos/checkerboard_000_frames/"
framePath = '../../camera_calibration_videos/checkerboard_000_frames/*.jpg'

imageToCheck = '../../camera_calibration_videos/checkerboard_000_frames/frame255.jpg'

'''
    Generate dataset from videos if dataset is not available.
    If dataset available, move on
'''
if len(os.listdir(outputPath)) == 0:
    print("Directory is empty. Generating Frames from Videos")
    count = cut.generateFramesFromVideo(inputPath1, outputPath, 0)
    _ = cut.generateFramesFromVideo(inputPath2, outputPath, count+1)
else:
    print("Dataset available")

'''
    if camera_calib_pickle.p available, no training required
    if not, train camera data and store data into camera_calib_pickle.p file
'''
if os.path.isfile('camera_calib_pickle.p'):
    print("Training not required")
else:
    print("Training Required")
    ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = cut.calibTraining(cbrow, cbcol, framePath)
    cut.writeCalibResults(ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints)

'''
    If camera_calib_pickle.p file exist, read the data from it
'''
ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = cut.readCalibResults()

print("Camera Calibration:\n ", ret)
print("\nCamera Matrix:\n ", mtx)
print("\nDistortion Parameters\n: ", dist)
print("\nRotation Vectors:\n ", rvecs)
print("\nTranslation Vectors:\n ", tvecs)
print("\nObject Points:\n ", objpoints)
print("\nImage Points:\n ", imgpoints)

img = cv.imread(imageToCheck)
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('calibresult.png', dst)
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error / len(objpoints)))
