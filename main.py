# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:02:28 2018

@author: zhangyinwei
"""
import numpy as np
import cv2
import time

print('[info]: loading video...')
cap = cv2.VideoCapture('video.mp4')

print('[info]: initialize parameters...')
# params for ShiTomasi corner detection

# Parameters for lucas kanade optical flow
feature_params = dict( maxCorners = 5000,
                       qualityLevel = 0.01,
                       minDistance = 1,
                       blockSize = 5 )

# params for ShiTomasi corner detection
lk_params = dict( winSize  = (10,10),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

print('[info]: define function')
#change the image to gray scale
def change2gray():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray

    
def drawFeaturePoints(image, pts):
    pts = np.int0(pts)
    for i in pts:
        x, y = i.flatten()
        cv2.circle(image,(x, y), 3, [0, 255, ], -1)
    cv2.imshow('frame',image)

initial_time = time.time()

first_frame , first_gray = change2gray()
first_pts = cv2.goodFeaturesToTrack(first_gray, mask = None, **feature_params)

second_frame, second_gray = change2gray()
second_pts, st, err = cv2.calcOpticalFlowPyrLK(first_gray, second_gray, first_pts, None, **lk_params)
print('[info]: initialize video writter')

mask = np.zeros_like(first_gray)
color = np.random.randint(0,255,(100,3))
print('[info]: start detection...')
stat = list()
for i in range(180):
    print('[info]: frame num {}'.format(i))
    # get the next frame
    third_frame, third_gray = change2gray()
    # get the feature points
    third_pts, st, err = cv2.calcOpticalFlowPyrLK(first_gray, third_gray, first_pts, None, **lk_params)

    # homography transformation of the feature points in different frames
    M12, k12 = cv2.findHomography(second_pts, first_pts, cv2.RANSAC,2)
    M13, k13 = cv2.findHomography(third_pts, first_pts, cv2.RANSAC,2)   
    
    dst12 = cv2.warpPerspective(second_gray, M12, (second_gray.shape[1],second_gray.shape[0]))
    dst13 = cv2.warpPerspective(third_gray, M13, (third_gray.shape[1],third_gray.shape[0]))

    # frame differencing
    diff = cv2.absdiff(dst12, dst13)

    # post processing
    thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    erosion = cv2.erode(thresh, kernel_1, iterations = 1)
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(9, 9))
    dilation = cv2.dilate(erosion, kernel_2, iterations = 3)

    # _, contours, hierarchy = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # num = 0
    # loc = list()
    
    # for c in contours:
    #     if cv2.contourArea(c) < 5000:
    #         continue
    #     #if cv2.contourArea(c) > 2000000:
    #     #    continue
    #     rect = cv2.minAreaRect(c)
    #     h = rect[1][1]
    #     w = rect[1][0]
    #     #if h > 50 and w > 20 and h < 100 and w < 100:
    #     num += 1
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     loc.append(box.tolist())
    #     cv2.drawContours(first_frame, [box], 0, 255, 2)
    cv2.imshow('frame', thresh)

    first_frame = second_frame
    second_frame = third_frame
    
    first_gray = second_gray
    second_gray = third_gray
    first_pts = cv2.goodFeaturesToTrack(first_gray, mask = None, **feature_params)
    second_pts, st, err = cv2.calcOpticalFlowPyrLK(first_gray, second_gray, first_pts, None, **lk_params)
    k = cv2.waitKey(30) & 0xff
    if k == 30:
        break

cv2.destroyAllWindows()
cap.release()
print('[info]: complete...total running time: {}'.format(time.time()-initial_time))

