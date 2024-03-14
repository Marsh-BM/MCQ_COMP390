import cv2
import numpy as np
import os
import torch
from torchvision import transforms
import csv

class MCQScanner:
    def __init__(self, image, model, device, save_dir='results_txt'):
        self.image = image
        self.model = model
        self.device = device
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def preprocess(self, image):
        """图像预处理以匹配模型的输入要求。"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((30, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).to(self.device)  # 增加批次维度，并移动到设备上
        
    def predict_question(self, question_img):
        # 定义索引到字母的映射关系
        idx_to_answer = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'None'}
        """使用模型预测单个题目的答案。"""
        question_img = self.preprocess(question_img)
        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 不计算梯度，加速推理
            outputs = self.model(question_img)
            _, predicted = torch.max(outputs, 1)
            predicted_label = idx_to_answer[predicted.item()]  # 将预测的索引转换为对应的字母
        return predicted_label
        
    def crop_and_save_questions(self, start_x, start_y, question_width, question_height, h_space, v_space, rows,
                            columns, save_dir,page_num):
        """
        遍历答题卡，截取并保存每个题目。
        参数:
        - start_x, start_y: 第一个题目的起始坐标(左上角)56,48。
        - question_width, question_height: 每个题目的宽度 146和高度 30。
        - h_space, v_space: 题目之间的水平 298和垂直间距 45。
        - rows, columns: 答题卡上题目的行数 30和列数 4。
        - save_dir: 保存截取的题目的目录。
        """
        results = []
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        csv_file_path = os.path.join('results_txt', f"results_page_{page_num}.csv")
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Page', 'Question Number', 'Row', 'Column', 'Prediction'])  # 写入表头

            for col in range(columns):
                i = 0
                for row in range(rows):
                    if row == 0:
                        y = start_y
                    x = start_x + col * (h_space)

                    if row % 5 == 0 and row != 0:
                        i = i + 1
                    y = start_y + row * (question_height) + i * (v_space)
                    print(f"X:{x}")
                    print(f"Y:{y}")

                    question_img = self.image[y:y + question_height, x:x + question_width]

                    # 引入模型开始计算
                    prediction = self.predict_question(question_img)
                    question_number = col * rows + row + 1
                    results.append({'page_num': page_num, 'question_number': question_number, 'prediction': prediction})

                    # # 保存结果到文本文件
                    # result_text = f"Page {page_num}, Question {question_number}, Row: {row + 1}, Column: {col + 1}, Prediction: {prediction}\n"
                    # with open(os.path.join(self.save_dir, f"results_page_{page_num}.txt"), 'a') as file:
                    #     file.write(result_text)

                    # 写入CSV
                    writer.writerow([page_num, question_number, row + 1, col + 1, prediction])

                    file_name = f"Page_{page_num}_question_{row + 1}_{col + 1}.png"
                    file_path = os.path.join(save_dir, file_name)
                    cv2.imwrite(file_path, question_img)

                    print(f"Saved {file_path}")
                
            # return results




# class MCQScanner:
#     def __init__(self, image):
#         self.image = image

#     def crop_and_save_questions(self, start_x, start_y, question_width, question_height, h_space, v_space, rows,
#                                 columns, save_dir,page_num):
#         """
#         遍历答题卡，截取并保存每个题目。
#         参数:
#         - start_x, start_y: 第一个题目的起始坐标（左上角）56,48。
#         - question_width, question_height: 每个题目的宽度 146和高度 30。
#         - h_space, v_space: 题目之间的水平 298和垂直间距 45。
#         - rows, columns: 答题卡上题目的行数 30和列数 4。
#         - save_dir: 保存截取的题目的目录。
#         """
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)

#         for col in range(columns):
#             i = 0
#             for row in range(rows):
#                 if row == 0:
#                     y = start_y
#                 x = start_x + col * (h_space)

#                 if row % 5 == 0 and row != 0:
#                     i = i + 1
#                 y = start_y + row * (question_height) + i * (v_space)
#                 print(f"X:{x}")
#                 print(f"Y:{y}")

#                 question_img = self.image[y:y + question_height, x:x + question_width]

#                 file_name = f"Page_{page_num}_question_{row + 1}_{col + 1}.png"
#                 file_path = os.path.join(save_dir, file_name)
#                 cv2.imwrite(file_path, question_img)

#                 print(f"Saved {file_path}")