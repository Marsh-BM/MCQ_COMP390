

# # from pdf_to_image import
# from mcq_scanner import MCQScanner
# from image_processing import deskew
#
#
#
#
# if __name__ == '__main__':
#     image_path = 'MCQ5.png'
#     image = cv2.imread(image_path)
#     image = cv2.resize(image,(2479,3508))
#     if image is None:
#         print(f"Error: Unable to load image '{image_path}'.")
#         exit()
#
#     scale_percent = 50
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     dim = (width, height)
#
#     resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#     corrected_image = deskew(resized_image)
#
#     corrected_image = cv2.resize(corrected_image,(1119,1193))
#
#     save_path_Answer = 'Pictures'
#     file_name_Answer = "Answer5.png"
#
#     save_path_Questions = 'Questions\MCQ5'
#     file_name_Answer = "Question.png"
#
#     start_x, start_y = 58, 48  # 第一个题目的起始坐标
#     question_width, question_height = 150, 30  # 每个题目的宽度和高度
#     h_space, v_space = 297, 45  # 题目之间的水平和垂直间距
#     rows, columns = 30, 4  # 答题卡上题目的行数和列数
#
#     scanner = MCQScanner(corrected_image)
#     scanner.crop_and_save_questions(start_x, start_y, question_width, question_height, h_space, v_space, rows, columns, save_path_Questions)
#
#     cv2.imshow('Corrected Image', corrected_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()  # Close the window when done

from pdf_to_image import pdf_processing
import os
import cv2
import torch
from cuda_train import Net

# def main():
#     # 定义PDF文件路径和输出文件夹路径
#     pdf_path = os.path.join("PDF_Document", "Blank.pdf")
#     output_folder = os.path.join("JPG_Document","Blank") #!!!!!!!!!!!!!!!!!修改

#     current_dir = os.getcwd()

#     # 确保输出文件夹存在
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # 调用PDF处理函数
#     pdf_processing(pdf_path, output_folder)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model_path = 'Blank_epoch=10_lr=0.01_de2_accuracy'  # 模型文件路径
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    pdf_path = os.path.join("PDF_Document", "Batman.pdf")
    output_folder = os.path.join("JPG_Document","TruthData") #!!!!!!!!!!!!!!!!!修改

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_processing(pdf_path, output_folder, model, device)


if __name__ == "__main__":
    main()
