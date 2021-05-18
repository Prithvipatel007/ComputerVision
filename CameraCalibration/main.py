import cv2 as cv

vid = cv.VideoCapture('../../camera_calibration_videos/checkerboard_000.h264')
success, image = vid.read()
count = 0
while success:
    cv.imwrite("../../camera_calibration_videos/checkerboard_000_frames/frame%d.jpg" % count,
               image)  # save frame as JPEG file
    success, image = vid.read()
    print('count : ' + str(count))
    count += 1
