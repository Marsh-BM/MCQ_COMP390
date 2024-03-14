import fitz  # PyMuPDF
import os
import numpy as np
import cv2
from image_processing import deskew
from mcq_scanner import MCQScanner

def pdf_processing(pdf_path, output_folder, model, device):
    # 打开PDF文件
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        # 获取页面
        page = doc.load_page(page_num)

        # 渲染页面为像素图。dpi 参数控制输出图片的质量。
        pix = page.get_pixmap(dpi=300)

        # 从pixmap提取图像数据
        img_data = pix.samples

        # 定义输出图片的文件名
        filename = f"{output_folder}/page_{page_num + 1}.png"

        # 保存图片到指定路径
        pix.save(filename)

        # 将图像数据转换为numpy数组，注意这里的reshape方法参数需要根据实际情况调整
        # pix.width, pix.height为图像宽度和高度，3代表颜色通道数(RGB)
        image = np.frombuffer(img_data, dtype=np.uint8).reshape(pix.height, pix.width, 3)

        # 获取MCQ答题卡区域
        scale_percent = 50
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        corrected_image = deskew(resized_image)
        corrected_image = cv2.resize(corrected_image, (1119, 1193))

        # 在保存之前，将颜色空间从RGB转换为BGR
        corrected_image = cv2.cvtColor(corrected_image, cv2.COLOR_RGB2BGR)

        save_path_Answer = 'Pictures\Batman'#!!!!!!!!!!!!!!!!!修改
        file_name_Answer = f"Answer_Page_{page_num + 1}.png"
        if not os.path.exists(save_path_Answer):
            os.makedirs(save_path_Answer)
        full_save_path = os.path.join(save_path_Answer, file_name_Answer)
        try:
            cv2.imwrite(full_save_path, corrected_image)
            print(f"Image saved successfully to '{full_save_path}'.")
        except Exception as e:
            print(f"Error saving image: {e}")

        save_path_Questions = f'Result_Questions\Batman'#!!!!!!!!!!!!!!!!!修改
        # file_name_Answer = "Question.png"

        start_x, start_y = 58, 48  # 第一个题目的起始坐标
        question_width, question_height = 150, 30  # 每个题目的宽度和高度
        h_space, v_space = 297, 45  # 题目之间的水平和垂直间距
        rows, columns = 30, 4  # 答题卡上题目的行数和列数

        scanner = MCQScanner(corrected_image, model, device, save_dir='results_txt')
        scanner.crop_and_save_questions(start_x, start_y, question_width, question_height, h_space, v_space, rows,
                                        columns, save_path_Questions,page_num)
        








