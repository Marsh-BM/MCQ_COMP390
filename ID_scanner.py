import cv2
import numpy as np
import os

def edge_detect(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # 灰度图像, 高斯核大小, 标准差
    # 边缘检测
    edged = cv2.Canny(blurred, 50, 150)
    return edged


# 轮廓检测
def find_contours(image):
    # 边缘检测
    edged = edge_detect(image)
    # 轮廓检测
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# 裁剪图像
def crop(rotated, contours, h, w):
        # 矩形四角坐标
    contours = find_contours(rotated)
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    x1, y1 = box[0]
    x2, y2 = box[2]
    
    if y1 < h - y2:
        rotated = cv2.rotate(rotated, cv2.ROTATE_180)
        x1, y1 = w - x1, h - y1
        x2, y2 = w - x2, h - y2

    # 裁剪出矩形答题区域
    # rotated = rotated[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
    # 裁剪出学号区域
    rotated = rotated[int(0.46*min(y1, y2)): int(0.94*min(y1, y2)),int(0.76*max(x1, x2)):int(0.98*max(x1, x2))]
    return rotated

# 旋转图像
def deskew(image):
    # 获取图像大小
    (h, w) = image.shape[:2]

    # 轮廓检测
    contours = find_contours(image)

    # 画出轮廓
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # 找到最大的轮廓
    c = max(contours, key=cv2.contourArea)

    # 矩形拟合
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # 旋转矩形, 窄边为水平
    angle = rect[2] # 获取旋转角度
    if rect[1][0] > rect[1][1]:
        angle = 90 + angle

    # 旋转图像
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0) # 获取旋转矩阵
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) # 旋转图像

    # 裁剪图像
    rotated = crop(rotated, contours, h, w)

    return rotated

def split_image(image_path, output_dir):
    # Read the image
    image = cv2.imread(image_path)
    # Get the image's height and width
    height, width = image.shape[:2]
    # Calculate the width of each part
    part_width = width // 10
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Split and save each part of the image
    for i in range(10):
        # Calculate the start and end x coordinates for each part
        start_x = i * part_width
        end_x = start_x + part_width
        # Crop the image
        cropped_image = image[:, start_x:end_x]
        # Save the cropped image
        output_path = os.path.join(output_dir, f'part_{i+1}.jpg')
        cv2.imwrite(output_path, cropped_image)



    
if __name__ == '__main__':
    # Read the image
    image = cv2.imread('Batman.png')    # 缩小图像大小
    scale_percent = 50 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


    # Deskew the image
    deskewed = deskew(image)

    # 裁剪图像

    cv2.imshow('Deskewed', deskewed)
    cv2.waitKey(0)

    # Save the image
    cv2.imwrite('test.jpg', deskewed)

    # Directory to save the split images
    output_dir = 'ID_test'

    # Split the image and save the parts
    split_image('test.jpg', output_dir)