from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# 构建参数并解析参数
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to the image to be scanned")
# args = vars(ap.parse_args())
args = {'image': 'images/receipt.jpg'}
# 第一步 ： 边缘检测
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500
orig = image.copy()
image = imutils.resize(image, height=500)
image = imutils.rotate(image, 90)

# 将图片转换为灰度图像，并检测边缘
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 高斯模糊消除高频噪声
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# 显示原始图像和边缘检测图像
print("STEP 1: Edge Detection")

cv2.imshow("Image", image)

cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()  # 清理窗口

# 第二步：查找轮廓
# 我们假设要扫面的纸张是矩形的，常见一个简单的启发式方法来构建文档扫面程序
# 假设图像中具有恰好四个点的最大轮廓（或者至少具有四个边缘），则将是我们要扫面的纸张

# 找到最大的轮廓线
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# 循环轮廓线
for c in cnts:
    # 估算轮廓
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 如果检测到有4 个点
    if len(approx == 4):
        screenCnt = approx
        break

# 显示图片边缘
print("STEP 2:Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("outLine", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 第三步：应用透视变换的阈值
# 通过四个点，应用透视转换，获得自上而下的鸟瞰图
warped = four_point_transform(imutils.rotate(orig, 90), screenCnt.reshape(4, 2)*ratio)

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped > T).astype("uint8") * 255

# 显示原始图片和扫面的图片
print("STEP 3: Apply perspective transform")
cv2.imshow("original", imutils.resize(image, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)


