import cv2
import numpy as np
import os
import torch
from torchvision import transforms
import csv
# def edge_detect(image):
#     # 转换为灰度图像
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # 高斯滤波
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0) # 灰度图像, 高斯核大小, 标准差
#     # 边缘检测
#     edged = cv2.Canny(blurred, 50, 150)
#     return edged


# # 轮廓检测
# def find_contours(image):
#     # 边缘检测
#     edged = edge_detect(image)
#     # 轮廓检测
#     contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours


# # 裁剪图像
# def crop(rotated, contours, h, w):
#         # 矩形四角坐标
#     contours = find_contours(rotated)
#     c = max(contours, key=cv2.contourArea)
#     rect = cv2.minAreaRect(c)
#     box = cv2.boxPoints(rect)
#     box = np.intp(box)
#     x1, y1 = box[0]
#     x2, y2 = box[2]
    
#     if y1 < h - y2:
#         rotated = cv2.rotate(rotated, cv2.ROTATE_180)
#         x1, y1 = w - x1, h - y1
#         x2, y2 = w - x2, h - y2

#     # 裁剪出矩形答题区域
#     answer_area = rotated[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
#     # 裁剪出学号区域
#     id_area = rotated[int(0.47*min(y1, y2)): int(0.93*min(y1, y2)),int(0.76*max(x1, x2)):int(0.98*max(x1, x2))]
#     return answer_area,id_area

# # 旋转图像
# def deskew(image):
#     # 获取图像大小
#     (h, w) = image.shape[:2]

#     # 轮廓检测
#     contours = find_contours(image)

#     # 画出轮廓
#     # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

#     # 找到最大的轮廓
#     c = max(contours, key=cv2.contourArea)

#     # 矩形拟合
#     rect = cv2.minAreaRect(c)
#     box = cv2.boxPoints(rect)
#     box = np.intp(box)
    
#     # 旋转矩形, 窄边为水平
#     angle = rect[2] # 获取旋转角度
#     if rect[1][0] > rect[1][1]:
#         angle = 90 + angle

#     # 旋转图像
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0) # 获取旋转矩阵
#     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) # 旋转图像

#     # 裁剪图像
#     answer_area, id_area = crop(rotated, contours, h, w)

#     return answer_area,id_area

class IDScanner:
    def __init__(self, image, model, device, save_dir):
    # def __init__(self, image, model, device):
        self.image = image # ID_
        self.model = model
        self.device = device
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def preprocess(self, image):
        """图像预处理以匹配模型的输入要求。"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((184, 30)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])
        return transform(image).unsqueeze(0).to(self.device)  # 增加批次维度，并移动到设备上
    
    def predict_digit(self, question_img):
        # 定义索引到字母的映射关系
        # idx_to_answer = {'0_file':0,'1_file': 1, '2_file': 2, '3_file': 3, '4_file':4, '5_file':5, '6_file':6, '7_file':7, '8_file':8, '9_file':9}
        idx_to_answer = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
        """使用模型预测单个题目的答案。"""
        question_img = self.preprocess(question_img)
        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 不计算梯度，加速推理
            outputs = self.model(question_img)
            _, predicted = torch.max(outputs, 1)
            predicted_ID = idx_to_answer[predicted.item()]  # 将预测的索引转换为对应的字母
        return predicted_ID

    def split_id(self, page_num):
    # def split_id(self):
        image = self.image
        predicted_id = '' 
        # Get the image`s height and width
        height, width = image.shape[:2]
        # Calculate the total spacing width (8 gaps * 3 pixels)
        total_spacing_width = 8 * 3
        # Adjust the total width available for the parts
        available_width = width - total_spacing_width
        # Calculate the width of each part
        part_width = available_width // 9
        # 
        output_dir = 'ID_test'
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Split and save each part of the image
        for i in range(9):
            # Calculate the start x coordinate for each part
            # Each part's start point is offset by the (i * part_width) plus (i * 3) for the gaps
            start_x = i * (part_width + 4)
            end_x = start_x + part_width
            # Crop the image
            cropped_image = self.image[:, start_x:end_x]

            predicted_digit = self.predict_digit(cropped_image)
            predicted_id += predicted_digit

            # Save the cropped image
            output_path = os.path.join(output_dir, f'{page_num}_part_{i+1}.jpg')
            cv2.imwrite(output_path, cropped_image)

        print(predicted_id)
        return predicted_id




    
# if __name__ == '__main__':
#     # Read the image
#     image = cv2.imread('id_train0.png')    # 缩小图像大小
#     # image = cv2.imread('Batman.png') 
#     scale_percent = 50 # percent of original size
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


#     # Deskew the image
#     deskewed = deskew(image)

#     # 裁剪图像

#     cv2.imshow('Deskewed', deskewed)
#     cv2.waitKey(0)

#     # Save the image
#     cv2.imwrite('test01.png', deskewed)

#     # Directory to save the split images
#     output_dir = 'ID_test'

#     # Split the image and save the parts
#     split_id('test01.png', output_dir)