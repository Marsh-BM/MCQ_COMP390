import cv2
import numpy as np
import os
import torch
from torchvision import transforms
import csv

class MCQScanner:
    def __init__(self, image, model, device):
        self.image = image
        self.model = model
        self.device = device
        self.idx_to_answer = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'None'}

    def preprocess(self, images):
        """批量图像预处理以匹配模型的输入要求。"""
        processed_images = []
        for image in images:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((30, 150)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),
            ])
            processed_image = transform(image).unsqueeze(0)  # 单个图像处理
            processed_images.append(processed_image)
        
        return torch.cat(processed_images, dim=0)  # 合并所有处理后的图像为一个批次

    def predict_questions(self, images):
        """批量使用模型预测多个图像的答案。"""
        images = self.preprocess(images).to(self.device)  # 预处理整个图像批次
        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 不计算梯度，加速推理
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            predicted_labels = [self.idx_to_answer[pred.item()] for pred in predicted]
        
        return predicted_labels

    def crop_and_save_questions(self, start_x, start_y, question_width, question_height, h_space, v_space, rows, columns, Question_path, page_num):
        cropped_images = []  # 用于存储裁剪的图像
        results = []

        if not os.path.exists(Question_path):
            os.makedirs(Question_path)

        for col in range(columns):
            i = 0
            for row in range(rows):
                if row == 0:
                    y = start_y
                x = start_x + col * (h_space)

                if row % 5 == 0 and row != 0:
                    i = i + 1
                y = start_y + row * (question_height) + i * (v_space)
                # print(f"X:{x}")
                # print(f"Y:{y}")

                cropped_image = self.image[y:y + question_height, x:x + question_width]
                cropped_images.append(cropped_image)  # 添加到列表而不是立即预测

        predicted_labels = self.predict_questions(cropped_images)  # 批量预测

        # 将每个问题的预测结果和相应的图像保存下来
        question_number = 0
        for col in range(columns):
            for row in range(rows):
                question_number += 1
                prediction = predicted_labels[question_number - 1]
                results.append({'page_num': page_num, 'question_number': question_number, 'prediction': prediction})
                
                # 保存裁剪的图像
                # file_name = f"Page_{page_num}_question_{question_number}.png"
                # file_path = os.path.join(Question_path, file_name)
                # cv2.imwrite(file_path, cropped_images[question_number - 1])

        return results






