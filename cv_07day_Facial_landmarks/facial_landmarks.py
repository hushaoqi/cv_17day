# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the face number
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)


'''
使用dlib和OpenCV来检测  图像中的面部标志。
面部标志用于定位和表示面部的显着区域，例如：
眼睛
眉毛
鼻子
口
下巴
面部标志已经成功应用于面部对齐，头部姿势估计，面部交换，眨眼检测等等。

检测面部标志是一个两步过程：
步骤＃1：在图像中本地化面部。使用OpenCV的内置Haar级联 or 应用预先训练的HOG +线性SVM物体探测器专门用于人脸检测任务 or 使用基于深度学习的算法进行人脸定位
步骤＃2：检测面部ROI上的关键面部结构。dlib库中包含的面部标志检测器是Kazemi和Sullivan（2014）的“回归树集合的一毫秒人脸对齐”的实现。
此方法首先使用：
1、在图像上标记面部地标的训练集。手动标记这些图像，指定围绕每个面部结构的区域的特定 （x，y）坐标。
更具体地说，是输入像素对之间距离的概率。
给定该训练数据，训练回归树的集合以直接从像素强度本身估计面部界标位置（即，没有发生“特征提取”）。
http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html
'''
