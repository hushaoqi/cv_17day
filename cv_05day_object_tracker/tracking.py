'''
步骤＃1： 使用计算机视觉技术检测彩球的存在。
步骤＃2： 在视频帧中移动时跟踪球，在移动时绘制其先前的位置。
'''

# import necessary packages
from collections import deque
# 将使用 deque，具有超快速附加和弹出以维护过去N（x，y）的列表我们视频流中球位置的。
# 保持这样的队列允许我们在跟踪时绘制球的“轨迹”。
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# 构建参数列表
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the(optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# 定义了HSV颜色空间中绿色的下边界和上边界
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])
# 如果未提供视频路径，请抓取参考对网络摄像头
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

# 允许相机或视频文件预热
time.sleep(2.0)

while True:
    # 抓住当前帧
    frame = vs.read()
    # 处理帧
    frame = frame[1] if args.get("video", False)else frame
    # 视频结束
    if frame is None: break
    # 重设帧大小，模糊处理，转到HSV颜色空间
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 构造一个颜色为“绿色”的掩码，然后执行一系列的侵蚀和膨胀去除了可能留在面具上的任何小斑点。
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 找出轮廓，初始化当前球的中心点(x, y)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cvts = imutils.grab_contours(cnts)
    center = None

    # 找到一个轮廓，然后处理
    if len(cnts) > 0:
        # cnts列表中  找到最大轮廓 ，
        # 计算blob的最小包围圆，然后计算中心（x，y）-坐标（即“质心”）
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5,(0, 0, 255), -1)
    # 更新质点队列
    pts.appendleft(center)

    # 循环遍历一组跟踪点
    for i in range(1, len(pts)):
        # 如果当前点或前一个点是 None
        # （表示在该给定帧中未成功检测到球），则我们忽略当前索引继续在pts上循环
        if pts[i - 1] is None or pts[i] is None:
            continue

        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2. line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # 如果按下 ”q“ , 则退出循环
    if key == ord('q'):
        break

# 如果我们没有使用视频文件，则停止camera 视频流
if not  args.get("video", False):
    vs.stop()
else:
    vs.release()
# 关闭所有窗口
cv2.destroyAllWindows()

