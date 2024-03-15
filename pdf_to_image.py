import fitz  # PyMuPDF
import os
import numpy as np
import cv2
from image_processing import deskew
from mcq_scanner import MCQScanner
from PIL import Image

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

def process_images(images,save_answer,save_questions,model,device,save_csv):
    # processed_images = []
    for page_num, img in images:
        resized_image = convert_and_resize_image(img)
        # 进行纠偏处理
        deskewed_image = deskew(resized_image)
        deskewed_image = cv2.resize(deskewed_image, (1119, 1193))

        # 保存处理后的图像
        save_path_Answer = save_answer
        file_name_Answer = f"Answer_Page_{page_num + 1}.png"
        save_image(deskewed_image, save_path_Answer, file_name_Answer)

        save_path_Questions = save_questions
        start_x, start_y = 60, 48  # 第一个题目的起始坐标
        question_width, question_height = 150, 30 # 每个题目的宽度和高度
        h_space, v_space = 297, 44  # 题目之间的水平和垂直间距
        rows, columns = 30, 4  # 答题卡上题目的行数和列数

        scanner = MCQScanner(deskewed_image,model,device,save_csv)
        scanner.crop_and_save_questions(start_x, start_y, question_width, question_height, h_space, v_space, rows,
                                        columns, save_path_Questions,page_num)






