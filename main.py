from pdf_to_image import pdf_to_png, process_images
import os
import cv2
import torch
from Questions_model import Questions_model
from ID_model import ID_model
import csv

def compare_answers(standard_csv_path, prediction_csv_path):
    standard_answers = {}
    with open(standard_csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            question_number, answer = row[1], row[4]
            standard_answers[question_number] = answer
    
    correct_count = 0
    total_count = 0
    errors = []  # 用于收集错误的题目信息
    with open(prediction_csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            question_number, prediction = row[2], row[3]
            if question_number in standard_answers:
                total_count += 1
                if prediction == standard_answers[question_number]:
                    correct_count += 1
                else:
                    errors.append((question_number, prediction, standard_answers[question_number]))  # 收集错误信息
    
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    print(f"Total Questions: {total_count}, Correct: {correct_count}, Accuracy: {accuracy:.2f}%")
    if errors:
        print("Incorrect Questions:")
        for error in errors:
            print(f"Question Number: {error[0]}, Prediction: {error[1]}, Correct Answer: {error[2]}")


def main(Qu_model_path,ID_model_path,pdf_path,out_path,save_answer,save_questions,save_ID,save_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    Qu_model = Questions_model().to(device)
    Qu_model.load_state_dict(torch.load(Qu_model_path, map_location=device))
    Qu_model.eval()

    # 加载ID识别模型
    id_model = ID_model().to(device)  
    id_model.load_state_dict(torch.load(ID_model_path, map_location=device))
    id_model.eval()

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # 将PDF转换为图像列表
    images = pdf_to_png(pdf_path)
    process_images(enumerate(images),save_answer,save_questions,save_ID,Qu_model,id_model,device,save_csv)
    

if __name__ == "__main__":
    Qu_model_path = 'lr0.0005_ep10'  # 模型文件路径
    ID_model_path = 'ID_lr0.0005_ep10'
    pdf_path = "PDF_Document/Batman.pdf"
    out_path = "JPG_Document/TruthData"
    save_answer = 'Answer_area/Batman'
    save_questions = 'Validation/Batman'
    save_ID = 'ID_middle'
    save_csv = 'results_txt/ID_Question.csv'
    main(Qu_model_path,ID_model_path, pdf_path, out_path, save_answer,save_questions,save_ID,save_csv)
    compare_answers('results_txt/Correct_Answer.csv','results_txt/ID_Question.csv')
