import fitz  # PyMuPDF
import os
import numpy as np
import cv2
from image_processing import deskew
from mcq_scanner import MCQScanner
from PIL import Image
import csv
from ID_scanner import IDScanner
import torch

def pdf_to_png(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        print(page_num)
    return images

def save_image(image, save_path, file_name):
    os.makedirs(save_path, exist_ok=True)  # 确保目录存在
    full_save_path = os.path.join(save_path, file_name)
    cv2.imwrite(full_save_path, image)
    print(f"Image saved successfully to '{full_save_path}'.")

def convert_and_resize_image(image):
    # 将Pillow图像转换为OpenCV图像
    open_cv_image = np.array(image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    # 获取图像的新尺寸
    scale_percent = 40  # 例子中的缩放百分比
    width = int(open_cv_image.shape[1] * scale_percent / 100)
    height = int(open_cv_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # 调整图像大小
    return cv2.resize(open_cv_image, dim, interpolation=cv2.INTER_AREA)

# 111111111111111111111111111111111111111111111111111111111111111111111111
def process_images(images,save_answer,save_questions,save_ID,Qu_model,ID_model,device,save_csv):
    rows_to_write = []
# 111111111111111111111111111111111111111111111111111111111111111111111111

# def process_images(images,save_answer,save_questions,Qu_model, device, save_csv):
    # processed_images = []
    for page_num, img in images:
        resized_image = convert_and_resize_image(img)
        # deskewed_image = deskew(resized_image)
        # 裁剪答题卡，获得答题区域和ID区域
        answer_area, id_area = deskew(resized_image)
        answer_area = cv2.resize(answer_area, (1119, 1193))
# 1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
        # 将ID裁剪输入进去
        scanner_ID = IDScanner(id_area, ID_model, device, save_ID)
        predicted_id=scanner_ID.split_id(page_num)
# 1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111  
        # 保存处理后的图像
        save_path_Answer = save_answer
        file_name_Answer = f"Answer_Page_{page_num + 1}.png"
        save_image(answer_area, save_path_Answer, file_name_Answer)

        save_path_Questions = save_questions
        start_x, start_y = 60, 48  # 第一个题目的起始坐标
        question_width, question_height = 150, 30 # 每个题目的宽度和高度
        h_space, v_space = 297, 44  # 题目之间的水平和垂直间距
        rows, columns = 30, 4  # 答题卡上题目的行数和列数

        scanner_questions = MCQScanner(answer_area,Qu_model,device,save_csv)
        predicted_questions=scanner_questions.crop_and_save_questions(start_x, start_y, question_width, question_height, h_space, v_space, rows,
                                        columns, save_path_Questions,page_num)
        
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 对于这一页的每个问题，写入它们的信息到CSV文件
        for question in predicted_questions:
            row = ([predicted_id, question['page_num'], question['question_number'], question['prediction']])
            rows_to_write.append(row)


            
    # 一次性写入所有行到CSV文件
    with open(save_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Page', 'Question', 'Prediction'])
        writer.writerows(rows_to_write)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!





# if __name__ == "__main__":
#     images = pdf_to_png('PDF_Document/test_data.pdf')
#     save_answer = 'Answer_area_3/Test'
#     save_questions = 'test_data_3/C'
#     Qu_model = ' '
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     save_csv = ' '
#     process_images(enumerate(images),save_answer=save_answer,save_questions=save_questions,Qu_model=Qu_model,device=device,save_csv=save_csv)

