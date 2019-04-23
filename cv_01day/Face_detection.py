import numpy as np
import argparse
import cv2

# # 构造参数解析并解析参数
# #  我们有三个必需的参数：
# ap = argparse.ArgumentParser()
# # - image  ：输入图像的路径。
# ap.add_argument("-i", "--image", required=True,help="path to input image")
# # - prototxt  ：Caffe原型文件的路径。
# ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
# # - model  ：预训练Caffe模型的路径。
# ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
# # 可选参数 - confidence，可以覆盖默认阈值0.5。
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#                 help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())


args = {"image": "rooster.jpg", "prototxt": "deploy.prototxt.txt", "model": "res10_300x300_ssd_iter_140000.caffemodel", "confidence": 0.5}
# 从磁盘加载我们的序列化模型并从我们的图像中创建一个blob：
print("[INFO] loading model...")
# 我们使用我们的- prototxt   和 - model 文件路径加载我们的模型 。我们将模型存储为 网络
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# 载入图片，并构建一个blob
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]  # 提取图像尺寸
# blobFramImage函数负责预处理，包括设置 blob 维度和规范化。
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300),(104.0, 177.0, 123.0))

# 面部检测
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# 遍历检测，框出检测到的面部
for i in range(0, detections.shape[2]):
    # 提取置信度
    confidence = detections[0, 0, i, 2]
    # 通过确保“置信度”大于最小置信度来过滤掉弱检测
    if confidence > args["confidence"]:
        # 计算边界框的（x，y）坐标
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # 绘制面部相关的的边界框
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
# 显示输出图像
cv2.imshow("Output", image)
cv2.waitKey(0)





