from pdf_to_image import pdf_to_png, process_images,process_multiple_pdfs
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

            # 记录答案，正确答案大写，错误答案小写，并保留题号，对于None使用'#'
            student_answers[student_id]['Answers'][question_number] = '#' if prediction == 'None' else (prediction.upper() if prediction == correct else prediction.lower())

            # 初始化学生分数记录
            if student_id not in student_scores:
                student_scores[student_id] = {'Total Score': 0, 'Parts': {p: 0 for p in parts_set}}

            # 更新得分
            if prediction == correct:
                student_scores[student_id]['Parts'][part] += weight
                student_scores[student_id]['Total Score'] += weight
            else:
                # 如果答案错误，打印学生ID，题号和正确答案
                print(f"Student ID: {student_id}, Question Number: {question_number}, Correct Answer: {correct}")

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

def student_to_txt(csv_file_path, output_dir):
    with open(csv_file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

        # Calculate the number of parts
        headers = reader.fieldnames
        part_headers = [h for h in headers if h.startswith('Part')]

        for row in reader:
            student_id = row['ID']
            student_file_path = os.path.join(output_dir, f"{student_id}.txt")

            with open(student_file_path, 'w', encoding='utf-8') as student_file:
                student_file.write(f"ID: {student_id}\n")
                student_file.write(f"Total Marks: {row['Total_Marks']}\n")
                
                # Write the scores for each part
                for part in part_headers:
                    student_file.write(f"{part}: {row[part]}\n")
                
                # Identify and write wrong answers
                wrong_answers = [f"{i}-{ans.upper()}" for i, ans in enumerate(row['Answer'], start=1) if ans.islower()]
                wrong_questions = ', '.join(wrong_answers) if wrong_answers else 'None'
                student_file.write(f"Wrong Questions: {wrong_questions}\n")












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
    # images = pdf_to_png(pdf_path)
    # process_images(enumerate(images),save_answer,save_questions,save_ID,Qu_model,id_model,device,save_csv)
    
    process_multiple_pdfs(folder_path=pdf_path, save_answer=save_answer,save_questions=save_questions,save_ID=save_ID,
                          Qu_model=Qu_model,ID_model=id_model,device=device,save_csv=save_csv)
 




if __name__ == "__main__":
    Qu_model_path = '4CN_bz8_lr0.0005_ep45_3'
    ID_model_path = 'ID_Model/ID_lr0.00005_ep30'
    # pdf_path = "PDF_Document/test_3.pdf"
    pdf_path = "PDF_Document/Multiple_PDF_test"
    out_path = "JPG_Document/TruthData"
    save_answer = 'Answer_area/test_new'
    save_questions = 'Validation/test_new'
    save_ID = 'ID_middle'
    save_csv = 'results_txt/ID_Question.csv'

    # 记录 main 函数的开始时间
    start_time_main = time.time()

    main(Qu_model_path,ID_model_path, pdf_path, out_path, save_answer,save_questions,save_ID,save_csv)
    compare_answers('results_txt/Correct_Answer.csv','results_txt/ID_Question.csv','results_txt/Student_Scores.csv')
    student_to_txt('results_txt/Student_Scores.csv','results_txt')
    end_time_main = time.time()
    print(f"main function took {end_time_main - start_time_main} seconds to run.")




def run_main_process(Qu_model_path, ID_model_path, pdf_path, out_path, save_answer, save_questions, save_ID, save_csv, csv_path, save_result):
    start_time_main = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Qu_model = Questions_model().to(device)
    Qu_model.load_state_dict(torch.load(Qu_model_path, map_location=device))
    Qu_model.eval()

    id_model = ID_model().to(device)  
    id_model.load_state_dict(torch.load(ID_model_path, map_location=device))
    id_model.eval()

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # images = pdf_to_png(pdf_path)
    # process_images(enumerate(images), save_answer, save_questions, save_ID, Qu_model, id_model, device, save_csv)
    process_multiple_pdfs(folder_path=pdf_path, save_answer=save_answer,save_questions=save_questions,save_ID=save_ID,
                          Qu_model=Qu_model,ID_model=id_model,device=device,save_csv=save_csv)
    compare_answers(csv_path, 'results_txt/ID_Question.csv', save_result)
    end_time_main = time.time()
    print(f"main function took {end_time_main - start_time_main} seconds to run.")