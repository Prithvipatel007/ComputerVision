import cv2 as cv
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt


def generateFramesFromVideo(inputPath, outputPath):
    vid = cv.VideoCapture(inputPath)  # input path as Video
    success, image = vid.read()
    count = 0
    while success:
        cv.imwrite(outputPath + "frame%d.jpg" % count,
                   image)  # save frame as JPEG file
        success, image = vid.read()
        print('count : ' + str(count))
        count += 1


def calibTraining(cbrow, cbcol, framePath):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0),...,(6,5,0)
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(framePath)

    for fname in images:
        print("Processing for image " + str(fname))
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (cbcol, cbrow), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (cbcol, cbrow), corners2, ret)

    print("Processing Camera Calibration")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints


def writeCalibResults(ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints):
    # Save the camera calibration results.
    calib_result_pickle = {"ret": ret,
                           "mtx": mtx,
                           "dist": dist,
                           "rvecs": rvecs,
                           "tvecs": tvecs,
                           "objpoints": objpoints,
                           "imgpoints": imgpoints}

    pickle.dump(calib_result_pickle, open("camera_calib_pickle.p", "wb"))

def readCalibResults():
    calib_result_pickle = pickle.load(open("camera_calib_pickle.p", "rb"))
    ret = calib_result_pickle["ret"]
    mtx = calib_result_pickle["mtx"]
    dist = calib_result_pickle["dist"]
    rvecs = calib_result_pickle["rvecs"]
    tvecs = calib_result_pickle["tvecs"]
    objpoints = calib_result_pickle["objpoints"]
    imgpoints = calib_result_pickle["imgpoints"]

    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints