import cv2 as cv
import numpy as np
import imutils
from matplotlib.pyplot import contour
from skimage.filters import threshold_local
import matplotlib.pyplot as plt


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0.0
    for i in contours:
        area = cv.contourArea(i)
        if area > 100.0:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


# Load the image using cv
img = cv.imread('../../../images/document2.jpg')
#cv.imshow('Document Original', img)

rows = img.shape[0]
cols = img.shape[1]

# convert image into grayscale
grayScaleImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurredImage = cv.GaussianBlur(grayScaleImage, (5, 5), 1)
edges = cv.Canny(blurredImage, 50, 150)

contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
print("Number of contours = " + str(len(contours)))

'''
Finding the Biggest Contour Points to find Corners of the document
'''
biggestContourVector, maxArea = biggestContour(contours)

'''
reorder the coordinates
'''
biggestContourVector = reorder(biggestContourVector)

cv.drawContours(img, biggestContourVector, -1, (0, 255, 0), 20)

actualArea = np.float32(biggestContourVector)
expectedArea = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
prepTransform = cv.getPerspectiveTransform(actualArea, expectedArea)
imgTransform = cv.warpPerspective(img, prepTransform, (cols, rows))

grayImgTransform = cv.cvtColor(imgTransform, cv.COLOR_BGR2GRAY)
#ret, thresh1 = cv.threshold(grayImgTransform, 139, 255, cv.THRESH_BINARY)

se=cv.getStructuringElement(cv.MORPH_RECT , (8,8))
bg=cv.morphologyEx(grayImgTransform, cv.MORPH_DILATE, se)
out_gray=cv.divide(grayImgTransform, bg, scale=255)
out_binary=cv.threshold(out_gray, 0, 255, cv.THRESH_OTSU)[1]

#imgAdaptiveThre = cv.adaptiveThreshold(out_binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 7)

cv.imshow('Binarized image', out_binary)

cv.waitKey(0)
cv.destroyAllWindows()
