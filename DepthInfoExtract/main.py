import os
import CalibUtils as cut
import cv2 as cv
import logging
import numpy as np

cbrow = 5
cbcol = 5
inputPath1 = '../../DepthInfoVideos/250_cm_distance.h264'
inputPath2 = '../../DepthInfoVideos/3D_metal_printer.h264'
inputPath3 = '../../DepthInfoVideos/schwalbe.h264'  
outputPath = "../../DepthInfoVideos/images/"
framePath = '../../DepthInfoVideos/images/*.jpg'

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
        logging.info("Directory is empty. Generating Frames from Videos")
        count = cut.generateFramesFromVideo(inputPath1, outputPath, 0)
        #count1 = cut.generateFramesFromVideo(inputPath2, outputPath, count)
        #count2 = cut.generateFramesFromVideo(inputPath3, outputPath, count1)
        #count3 = cut.generateFramesFromVideo(inputPath4, outputPath, count2)
    else:
        logging.info("Dataset available")

    '''
        if filename.p available, no training required
        if not, train camera data and store data into camera_calib_pickle.p file
    '''
    if os.path.isfile(filename):
        logging.info("Training not required")
    else:
        logging.info("Training Required")
        ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = cut.calibTraining(cbrow, cbcol, framePath)
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
        logging.info("Images already Undistorted")

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    logging.info("total error: {}".format(mean_error / len(objpoints)))
    return 0

if __name__ == "__main__":
    executeCameraCalibration(originalPickleFile)
    ''' Calculate the distance '''
    ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = cut.readCalibResults(originalPickleFile)
    logging.info('Computing PNP')
    ret1, rvec1, tvec1 = cv.solvePnP( objpoints,imgpoints , mtx, dist)
    print(tvec1)




