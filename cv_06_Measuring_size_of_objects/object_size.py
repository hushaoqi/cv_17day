# import necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
    return ((ptA[0] +ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# 构建参数列表并解析
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image",required=True,
#                 help="path to the input image")
# ap.add_argument("-w", "--width", type=float, required=True,
#                 help="width of left-most object in the image (in inches)")
# args = vars(ap.parse_args())

args = {"image": "images/example_01.png", "width": 0.955}

# 导入图片，转化为灰度图像，并做模糊平滑处理
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# 进行边缘检测以及扩张+侵蚀，以封闭边缘图中边缘之间的任何间隙
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# 找到与我们的边缘图中的对象相对应的轮廓
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# 对这些轮廓进行排序
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# 检查每个轮廓
for c in cnts:
    # 如果轮廓太小，则忽略
    if cv2.contourArea(c) < 100:
        continue
    # 计算图像的旋转边界框
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # 将旋转的边界框坐标排列 在左上角，右上角，右下角和左下角
    box = perspective.order_points(box)
    # 以绿色绘制对象的轮廓
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    # 以小的红色圆圈绘制边界框矩形的顶点
    for (x, y) in box:
        cv2.circle(orig,(int(x), int(y)), 5, (0, 0, 255), -1)

    # 计算出一系列中点
    (tl, tr, br, bl) = box
    # 计算左上角和右上角之间的中点,然后是右下角之间的中点。
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    # 分别计算左上角+左下角和右上角+右下角之间的中点
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # 在我们的图像上绘制蓝色中点，然后用紫色线连接中点。
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # 画出中间点连线
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)

    # 计算我们的中点集之间的欧几里德距离
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    # 看看我们的 pixelsPerMetric变量已经初始化，
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]

    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # draw the object sizes on the image
    cv2.putText(orig, "{:.1f}in".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}in".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    # show the output image
    cv2.imshow("Image", orig)
    cv2.waitKey(0)