import fitz  # PyMuPDF
import os
import numpy as np
import cv2
from image_processing import deskew
from mcq_scanner import MCQScanner
from PIL import Image
import csv
from ID_scanner import IDScanner
import glob

def pdf_to_png(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append((page_num,img))
        print(page_num)
    return images

def save_image(image, save_path, file_name):
    os.makedirs(save_path, exist_ok=True)  
    full_save_path = os.path.join(save_path, file_name)
    cv2.imwrite(full_save_path, image)
    print(f"Image saved successfully to '{full_save_path}'.")

def convert_and_resize_image(image):
    # Convert Pillow images to OpenCV images
    open_cv_image = np.array(image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    # Resize the image
    scale_percent = 40  
    width = int(open_cv_image.shape[1] * scale_percent / 100)
    height = int(open_cv_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(open_cv_image, dim, interpolation=cv2.INTER_AREA)


def process_images(images,save_answer,save_questions,Qu_model,ID_model,device):
    rows_to_write = []

    for page_num, img in images:
        resized_image = convert_and_resize_image(img)
        # deskewed_image = deskew(resized_image)
        # Cut the answer key to get the answer area and ID area
        answer_area, id_area = deskew(resized_image)
        answer_area = cv2.resize(answer_area, (1119, 1193))

        # Enter the ID crop in
        scanner_ID = IDScanner(id_area, ID_model, device)
        predicted_id=scanner_ID.split_id(page_num)  

        # Saving processed images
        # save_path_Answer = save_answer
        # file_name_Answer = f"Answer_Page_{page_num + 1}.png"
        # save_image(answer_area, save_path_Answer, file_name_Answer)

        save_path_Questions = save_questions
        start_x, start_y = 60, 48  
        question_width, question_height = 150, 30 
        h_space, v_space = 297, 44  
        rows, columns = 30, 4  

        scanner_questions = MCQScanner(answer_area,Qu_model,device)
        predicted_questions=scanner_questions.crop_and_save_questions(start_x, start_y, question_width, question_height, h_space, v_space, rows,
                                        columns, save_path_Questions,page_num)
        
        #  For each question on this page, write their information to a CSV file
        for question in predicted_questions:
            row = ([predicted_id, question['page_num'], question['question_number'], question['prediction']])
            rows_to_write.append(row)

    return rows_to_write




# Main function to process multiple PDFs and output a single CSV
def process_multiple_pdfs(folder_path, save_answer, save_questions,Qu_model, ID_model, device, predict_csv):
    initialize_csv(predict_csv)  # Initialize the CSV file
    all_rows_to_write = []
    # Use glob pattern matching to find all PDF files
    pdf_paths = glob.glob(os.path.join(folder_path, '*.pdf'))
    for pdf_path in pdf_paths:
        images = pdf_to_png(pdf_path)
        rows_to_write = process_images(images, save_answer, save_questions, Qu_model, ID_model, device)
        all_rows_to_write.extend(rows_to_write)
    # Uniformly write to CSV after processing all PDFs
    with open(predict_csv, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_rows_to_write)
    

def initialize_csv(csv_path):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Page Number', 'Question Number', 'Prediction'])





