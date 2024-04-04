from pdf_to_image import pdf_to_png, process_images
import os
import cv2
import torch
from Questions_model import Questions_model
from ID_model import ID_model
import csv
import time

def compare_answers(standard_csv_path, prediction_csv_path, output_csv_path):
    # 读取标准答案
    standard_answers = {}
    parts_set = set()  # 用于保存所有独特的部分名称
    with open(standard_csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            question_number, answer, part, weight = int(row[1]), row[4], row[5], int(row[6])
            standard_answers[question_number] = {'answer': answer, 'part': part, 'weight': weight}
            parts_set.add(part)  # 添加部分到集合中

    # 初始化学生得分字典和答案字典
    student_scores = {}
    student_answers = {}

    # 读取预测文件并比较答案
    with open(prediction_csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            student_id, question_number, prediction, page = row[0], int(row[2]), row[3], row[1]
            correct = standard_answers[question_number]['answer']
            part = standard_answers[question_number]['part']
            weight = standard_answers[question_number]['weight']

            # 初始化学生答案记录
            if student_id not in student_answers:
                student_answers[student_id] = {'Page': page, 'Answers': {}}

            # 记录答案，正确答案大写，错误答案小写，并保留题号
            student_answers[student_id]['Answers'][question_number] = prediction.upper() if prediction == correct else prediction.lower()

            # 初始化学生分数记录
            if student_id not in student_scores:
                student_scores[student_id] = {'Total Score': 0, 'Parts': {p: 0 for p in parts_set}}

            # 更新得分
            if prediction == correct:
                student_scores[student_id]['Parts'][part] += weight
                student_scores[student_id]['Total Score'] += weight

    # 将结果写入新的CSV文件
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        headers = ['ID', 'Page', 'Answer'] + sorted(parts_set) + ['Total_Marks']
        writer.writerow(headers)
        
        # 写入每个学生的得分和带序号的答案
        for student_id, data in student_answers.items():
            sorted_answers = sorted(data['Answers'].items())
            answer_str = ' '.join([f"{i}-{a}" for i, a in sorted_answers])  # 生成带序号的答案字符串
            row = [student_id, data['Page'], answer_str]
            row += [student_scores[student_id]['Parts'][p] for p in sorted(parts_set)]  # 按顺序添加每个部分的得分
            row.append(student_scores[student_id]['Total Score'])
            writer.writerow(row)
            


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

def run_main_process(Qu_model_path, ID_model_path, pdf_path, out_path, save_answer, save_questions, save_ID, save_csv, csv_path, save_result):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Qu_model = Questions_model().to(device)
    Qu_model.load_state_dict(torch.load(Qu_model_path, map_location=device))
    Qu_model.eval()

    id_model = ID_model().to(device)  
    id_model.load_state_dict(torch.load(ID_model_path, map_location=device))
    id_model.eval()

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    images = pdf_to_png(pdf_path)
    process_images(enumerate(images), save_answer, save_questions, save_ID, Qu_model, id_model, device, save_csv)

    compare_answers(csv_path, 'results_txt/ID_Question.csv', save_result)





if __name__ == "__main__":
    Qu_model_path = 'Question_Model/lr0.0005_ep10'  # 模型文件路径
    ID_model_path = 'ID_Model/ID_lr0.00005_ep30'
    pdf_path = "PDF_Document/test_3.pdf"
    out_path = "JPG_Document/TruthData"
    save_answer = 'Answer_area/test_new'
    save_questions = 'Validation/test_new'
    save_ID = 'ID_middle'
    save_csv = 'results_txt/ID_Question.csv'

    # 记录 main 函数的开始时间
    start_time_main = time.time()

    main(Qu_model_path,ID_model_path, pdf_path, out_path, save_answer,save_questions,save_ID,save_csv)
    compare_answers('results_txt/Correct_Answer.csv','results_txt/ID_Question.csv','results_txt/Student_Scores.csv')

    end_time_main = time.time()
    print(f"main function took {end_time_main - start_time_main} seconds to run.")



# def compare_answers(standard_csv_path, prediction_csv_path):
#     # 读取标准答案
#     standard_answers = {}
#     with open(standard_csv_path, 'r') as file:
#         reader = csv.reader(file)
#         next(reader)
#         for row in reader:
#             question_number, answer, part, weight = int(row[1]), row[4], row[5], int(row[6])
#             standard_answers[question_number] = {'answer': answer, 'part': part, 'weight': weight}

#     # 初始化学生得分字典
#     student_scores = {}

#     # 读取预测文件并比较答案
#     with open(prediction_csv_path, 'r') as file:
#         reader = csv.reader(file)
#         next(reader)
#         for row in reader:
#             student_id, question_number, prediction = row[0], int(row[2]), row[3]
#             if student_id not in student_scores:
#                 student_scores[student_id] = {'Total Score': 0, 'Parts': {}}
#             # 获取问题的部分和权重
#             part = standard_answers[question_number]['part']
#             weight = standard_answers[question_number]['weight']
#             # 初始化该部分的得分
#             if part not in student_scores[student_id]['Parts']:
#                 student_scores[student_id]['Parts'][part] = {'Score': 0, 'Total': 0}
#             # 更新得分
#             if prediction == standard_answers[question_number]['answer']:
#                 student_scores[student_id]['Parts'][part]['Score'] += weight
#                 student_scores[student_id]['Total Score'] += weight
#             student_scores[student_id]['Parts'][part]['Total'] += weight

#     # 打印每个学生的部分得分和总得分
#     for student_id, scores in student_scores.items():
#         print(f"Student ID: {student_id}")
#         for part, part_scores in scores['Parts'].items():
#             print(f"  {part} - Score: {part_scores['Score']} / {part_scores['Total']}")
#         print(f"Total Score: {scores['Total Score']}\n")