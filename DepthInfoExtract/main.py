import cv2 as cv


def generateFramesFromVideo(inputPath, outputPath, cnt):
    vid = cv.VideoCapture(inputPath)  # input path as Video
    success, image = vid.read()
    count = cnt
    while success:
        cv.imwrite(outputPath + "frame%d.jpg" % count,
                   image)  # save frame as JPEG file
        success, image = vid.read()
        print('count : ' + str(count))
        count += 1
    return count


def executeCameraCalibration():
    return 0


inputPath1 = '../../DepthInfoVideos/Vespa_04.h264'
outputPath = '../../DepthInfoVideos/Vespa_04/'

count = generateFramesFromVideo(inputPath1, outputPath, 0)
