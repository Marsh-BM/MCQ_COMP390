import cv2
import numpy as np
import os
import torch
from torchvision import transforms
import csv


class IDScanner:
    def __init__(self, image, model, device, save_dir):
    # def __init__(self, image, model, device):
        self.image = image # ID_
        self.model = model
        self.device = device
        self.save_dir = save_dir
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

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
        output_dir = 'ID_middle'
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