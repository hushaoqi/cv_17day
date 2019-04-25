'''
目标是使用Python和OpenCV构建一个气泡表扫描程序和测试分级程序。

为实现这一目标，我们的实施需要满足以下7个步骤：

步骤1：检测图像中的检查。
步骤＃2：应用透视变换以提取自上而下的鸟瞰视图。
步骤＃3：从透视变换检查中提取气泡集（即可能的答案选项）。
步骤＃4：将问题/气泡排序成行。
步骤＃5：确定每行的标记（即“冒泡”）答案。
步骤＃6：在我们的答案键中查找正确的答案，以确定用户的选择是否正确。
第7步：重复考试中的所有问题。
'''
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# 构建参数列表bingjiex
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to the input image")
# args = vars(ap.parse_args())
args = {"image": "images/test_02.png"}

# 定义题目与正确答案的映射表
ANSWER_KEY = {0:1, 1:4, 2:0, 3:3, 4:1}
# 问题1：  B
# 问题2：  E
# 问题3：  A
# 问题4：  D
# 问题5：  B

# 预处理输入图像，转化为灰度图像，做平滑处理，检测边缘
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# 获得文档的自上而下的鸟瞰视图
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

if len(cnts) > 0:
    # 根据轮廓大小降次排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # 循环遍历排序的轮廓
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)

        # 如果轮廓有四个角点，则可以假设找到了目标文档
        if len(approx) == 4:
            docCnt = approx
            break
    # 应用透视变换来获得文档的自上而下的鸟瞰图
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# 二值化，阈值处理
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# 注意图像的  背景 是  黑色的，而  前景是  白色的。
#
# 这种二值化将允许我们再次应用轮廓提取技术来查找考试中的每个气泡：
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

for c in cnts:
    # 计算轮廓的边界框，然后使用  bounding box派生宽高比
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # 为了将轮廓区域视为气泡，该区域应：
    # 足够宽和高（在这种情况下，两个维度中至少20个像素）。
    # 宽高比大约等于1
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)
# 霍夫圆圈不能很好地处理轮廓中的变形 - 在这种情况下，圆检测将完全失败
questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
correct = 0

# 每一个问题有5个可能的答案，循环遍历
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
    bubbled = None
# 给定一排气泡，下一步是确定填充了哪个气泡。
    for (j, c) in enumerate(cnts):
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    color = (0, 0, 255)
    k = ANSWER_KEY[q]

    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1
        print(correct)
    cv2.drawContours(paper, [cnts[k]], -1, color, 3)

# 处理考试评分并将结果显示在我们的屏幕上
score = (correct / 5.0) * 100


print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)
