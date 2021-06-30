import os
import CalibUtils as cut
import cv2 as cv
import logging
import numpy as np
import math
from numpy import linalg as LA

cbrow = 6
cbcol = 6
inputPath1 = '../../DepthInfoVideos/250_cm_distance.h264'
inputPath2 = '../../DepthInfoVideos/3D_metal_printer.h264'
inputPath3 = '../../DepthInfoVideos/schwalbe.h264'  
outputPath = "../../DepthInfoVideos/images/"
framePath = '../../DepthInfoVideos/images/*.jpg'
undistortedFramePath = '../../DepthInfoVideos/undistorted_images/*.png'

imageToCheck = '../../DepthInfoVideos/images/*.jpg'
undistortOutputPath = '../../DepthInfoVideos/undistorted_images/'
originalPickleFile = "camera_calib_pickle.p"
undistortedPickleFile = "undistorted_camera_calib_pickle.p"

def executeCameraCalibration(filename):
    '''
        Generate dataset from videos if dataset is not available.
        If dataset available, move on
    '''
    if len(os.listdir(outputPath)) == 0:
        print("Directory is empty. Generating Frames from Videos")
        count = cut.generateFramesFromVideo(inputPath1, outputPath, 0)
        #count1 = cut.generateFramesFromVideo(inputPath2, outputPath, count)
        #count2 = cut.generateFramesFromVideo(inputPath3, outputPath, count1)
        #count3 = cut.generateFramesFromVideo(inputPath4, outputPath, count2)
    else:
        print("Dataset available")

    '''
        if filename.p available, no training required
        if not, train camera data and store data into camera_calib_pickle.p file
    '''
    if os.path.isfile(filename):
        print("Training not required")
    else:
        print("Training Required")
        ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = cut.calibTraining(cbrow, cbcol, framePath, False)
        cut.writeCalibResults(ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, filename)

    '''
        If camera_calib_pickle.p file exist, read the data from it
    '''
    ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = cut.readCalibResults(filename)

    '''
    print("Camera Calibration:\n ", ret)
    print("\nCamera Matrix:\n ", mtx)
    print("\nDistortion Parameters\n: ", dist)
    print("\nRotation Vectors:\n ", rvecs)
    print("\nTranslation Vectors:\n ", tvecs)
    print("\nObject Points:\n ", objpoints)
    print("\nImage Points:\n ", imgpoints)
    '''
    if len(os.listdir(undistortOutputPath)) == 0:
        cut.undistort(imageToCheck, mtx, dist, undistortOutputPath)
    else:
        print("Images already Undistorted")

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))
    return 0

def executeCameraCalibrationUndistorted(filename):
    if os.path.isfile(filename):
        print("Training not required")
    else:
        print("Training Required")
        ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = cut.calibTraining(cbrow, cbcol, undistortedFramePath, True)
        cut.writeCalibResults(ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, filename)

if __name__ == "__main__":
    executeCameraCalibration(originalPickleFile)
    executeCameraCalibrationUndistorted(undistortedPickleFile)
    ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = cut.readCalibResults(undistortedPickleFile)
    # Distance
    p1 = np.array([700, 420, 1])
    p2 = np.array([812, 420, 1])
    v1 = np.matmul(LA.inv(mtx), p1)
    v2 = np.matmul(LA.inv(mtx), p2)
    theta = math.acos(np.dot(v1, v2) / (LA.norm(v1) * LA.norm(v2)))
    distance = 25 / math.tan(theta / 2)
    print(" Distance : " + str(distance))

    # Height
    h1 = np.array([684, 215, 1])
    h2 = np.array([684, 672, 1])
    v11 = np.matmul(LA.inv(mtx), h1)
    v12 = np.matmul(LA.inv(mtx), h2)
    theta1 = math.acos(np.dot(v11, v12) / (LA.norm(v11) * LA.norm(v12)))
    height = 2 * (distance * math.tan(theta1 / 2))
    print(" Height : " + str(height))

    # Width
    w1 = np.array([685, 421, 1])
    w2 = np.array([919, 421, 1])
    v21 = np.matmul(LA.inv(mtx), w1)
    v22 = np.matmul(LA.inv(mtx), w2)
    theta2 = math.acos(np.dot(v21, v22) / (LA.norm(v21) * LA.norm(v22)))
    width = 2 * (distance * math.tan(theta2 / 2))
    print(" Width : " + str(width))




