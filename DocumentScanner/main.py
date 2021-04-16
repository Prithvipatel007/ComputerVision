import cv2 as cv
import numpy as np
import imutils
from skimage.filters import threshold_local
import matplotlib.pyplot as plt

# Load the image using cv

img = cv.imread('../../../images/document.jpg')
# cv.imshow('Document Original', img)

# convert image into grayscale
grayScaleImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('GrayScaled Image', grayScaleImage)

# find edges

edges = cv.Canny(grayScaleImage, 110, 200)
#cv.imshow('Image with edges', edges)

cv.waitKey(0)
cv.destroyAllWindows()