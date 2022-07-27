from PIL import Image
import pytesseract
import cv2
import os

preprocess = 'blur' #thresh

# 读入scan.py的输出结果图像：scan.jpg
image = cv2.imread('scan.jpg')
# 转灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 将一幅灰度图二值化
if preprocess == "thresh":
    gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# 实现中值滤波
if preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)
    
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)
    
text = pytesseract.image_to_string(Image.open(filename))
print(text)
os.remove(filename)

cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)                                   
