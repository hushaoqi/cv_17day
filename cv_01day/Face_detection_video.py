# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# # 构建参数并解析
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
#                 help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
#                 help="path to caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#                 help="minimum probability to fiter weak detections")
# args = vars(ap.parse_args())
# # 这次我们没有 -image 路径参数。我们将使用我们的网络摄像头视频源。

args = {"image": "rooster.jpg", "prototxt": "deploy.prototxt.txt", "model": "res10_300x300_ssd_iter_140000.caffemodel", "confidence": 0.5}


# 加载模型并初始化视频流
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# 初始化视频流并允许摄像机传感器预热
print("[INFO] starting video stream...")
# 初始化一个 VideoStream  对象，
# 指定索引为零的摄像机作为源（通常这将是您的笔记本电脑的内置摄像头或您的桌面的第一个检测到的摄像头）
# 如果要解析视频文件（而不是视频流），请替换 VideoStream 类 为 FileVream 类
vs = VideoStream(src=0).start()
# 然后我们让相机传感器预热2秒钟
time.sleep(4.0)

# 循环框架并使用OpenCV计算面部检测
while True:
    # 从线程视频流中获取帧并调整其大小,最大宽度为400像素
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # 获取帧尺寸并将其转换为blob(二进制大对象(BLOBS BinaryLargeOBjects))
    (h, w) = frame.shape[: 2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    # 过网络传递blob，得到检测结果
    net.setInput(blob)
    detections = net.forward()

    # 现在可以遍历检测，与置信度阈值进行比较，并在屏幕上绘制面框+置信度值
    for i in range(0, detections.shape[2]):
        # 提取置信度
        confidence = detections[0, 0, i, 2]
        # 通过确保“置信度”大于最小置信度来过滤掉弱检测
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 绘制面部相关的的边界框
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # 现在已经绘制了我们的OpenCV面部检测，让我们在屏幕上显示框架并等待按键
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # 如果 按下‘q'键， 退出循环
    if key == ord("q"):
        break

# 做一些清理工作
cv2.destroyAllWindows()
vs.stop()
