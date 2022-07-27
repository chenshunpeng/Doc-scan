# 导入工具包
import numpy as np
import argparse
import cv2

# 设置命令行参数
# 构造参数解析并解析参数
# we instantiate the ArgumentParser object as ap（实例化）
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default='./images/pic.jpg'
				, required = False, help = "Path to the image to be scanned")
args = vars(ap.parse_args())

def order_points(pts):
	# 一共4个坐标点
	rect = np.zeros((4, 2), dtype = "float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):
	# 获取输入坐标点
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# 计算输入的w和h值
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 变换后对应坐标位置
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# 计算变换矩阵
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回变换后结果
	return warped

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation=inter)
	return resized

# 读取输入
image = cv2.imread(args["image"])
# 图像缩放，坐标也会相同变化
ratio = image.shape[0] / 500.0
orig = image.copy()
image = resize(orig, height = 500)

# 预处理

# 转灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 高斯滤波，去除噪音点
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# 边缘检测
edged = cv2.Canny(gray, 75, 200)

# 展示预处理结果
print("STEP 1: 边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 轮廓检测
# 在 Opencv4 中cv2.findContour()仅返回 2 个值：contours, hierachy，所以在这里用[0]得到第一个值
# 可借鉴这个网址 https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

# cv.findContours（）函数中有三个参数，第一个是源图像，第二个是轮廓检索模式，第三个是轮廓近似方法。
# 它输出轮廓和层次结构。Contours是图像中所有轮廓的Python列表。每个单独的轮廓都是对象边界点的 （x，y） 坐标的 Numpy 数组
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
# 对轮廓按照面积从大到小排序，取前5个（先从小到大排序，之后取reverse翻转）
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
# 遍历轮廓

# 对screenCnt初始化，不然可能会有警告
screenCnt = [[0,0], [255,0], [255,255], [0,255]]

for c in cnts:
	# 计算轮廓近似
	peri = cv2.arcLength(c, True)

	# cv2.approxPolyDP()的主要功能是把一个连续光滑曲线折线化，之后多边形逼近
	# c表示输入的点集
	# 其中第二个参数epsilon的作用：double epsilon：判断点到相对应的line segment的距离的阈值
	# （距离大于此阈值则舍弃，小于此阈值则保留，epsilon越小，折线的形状越“接近”曲线。）
	# True表示封闭的
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# 因为是文本行，返回的框至少应该是个四边形，只要找到那个最大的四边形，就可以退出了
	if len(approx) == 4:
		screenCnt = approx
		break

# 展示结果
print("STEP 2: 获取轮廓")
# 第一个参数image表示绘制的目标图像
# 第二个参数contours表示输入的轮廓组
# 第三个参数contourIdx指明画第几个轮廓****，如果该参数为负值，则画全部轮廓，
# 第四个参数color为轮廓的颜色，
# 第五个参数thickness为轮廓的线宽，如果为负值表示填充轮廓内部，
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 透视变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# 灰度，二值处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]

# 把ref写入scan.jpg
cv2.imwrite('scan.jpg', ref)


# 修改图片大小，同时图像逆时针旋转90度

# 获取图片，修改一下图片的大小
img = cv2.imread("scan.jpg")
# 注意需要制定返回值为img2，不能没有返回值
img2 = cv2.resize(img, (900, 600))
cv2.imshow("temp", img2)
cv2.waitKey(0)
# 对图片进行旋转
# 方法一
# img90 = np.rot90(img2)
# 方法二
# 绕任意点旋转
# 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
M = cv2.getRotationMatrix2D((450, 450), 90, 1)
# 仿射变化
# 第三个参数：输入图像的大小
img90 = cv2.warpAffine(img2, M, img2.shape[:2])
# (600, 900)与img2.shape[:2]等价
# img90 = cv2.warpAffine(img2, M, (600, 900))
cv2.imwrite('scan.jpg', img90)
# cv2.imshow("rotate", img90)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 展示结果
print("STEP 3: 变换")
# cv2.imshow("Original", resize(orig, height = 650))
cv2.imshow("Scanned", resize(img90, height = 650))
cv2.waitKey(0)