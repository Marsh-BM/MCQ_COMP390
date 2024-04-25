from pdf_to_image import process_multiple_pdfs
import os
import torch
from Questions_model import Questions_model
from ID_model import ID_model
import csv
import time
import pandas as pd

def convert_csv_format(input_csv_path, output_csv_path):
    # 读取原始 CSV 文件
    df = pd.read_csv(input_csv_path)
    
    # 选择必要的列并重命名（如果需要的话）
    new_df = df[['Page Number', 'Question Number', 'Prediction']]
    
    # 由于你的新 CSV 文件格式还需要 Row 和 Column 等列，我们可以添加这些列并为它们设置默认值
    new_df['Part'] = 'Part 1'  # 假设所有问题都属于 Part 1
    new_df['Weight'] = 1  # 假设所有问题的权重都是 1
    
    # 重命名列以符合目标 CSV 的格式要求
    new_df.rename(columns={'Page Number': 'Page', 'Question Number': 'Question'}, inplace=True)
    
    # 保存转换后的数据到新的 CSV 文件
    new_df.to_csv(output_csv_path, index=False)
    print(f"Converted CSV saved to {output_csv_path}")


def compare_answers(standard_csv_path, prediction_csv_path, output_csv_path):
    # 读取标准答案
    standard_answers = {}
    parts_set = set()  # 用于保存所有独特的部分名称
    with open(standard_csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            question_number, answer, part, weight = int(row[1]), row[2], row[3], int(row[4])
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
            student_answers[student_id]['Answers'][question_number] = '-' if prediction == 'None' else (prediction.upper() if prediction == correct else prediction.lower())

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
            answer_str = ' '.join([f"{i}.{a}" for i, a in sorted_answers])  # 生成带序号的答案字符串
            row = [student_id, data['Page'], answer_str]
            row += [student_scores[student_id]['Parts'][p] for p in sorted(parts_set)]  # 按顺序添加每个部分的得分
            row.append(student_scores[student_id]['Total Score'])
            writer.writerow(row)

def main(Qu_model_path,ID_model_path,pdf_path,out_path,save_answer,save_questions,predict_csv):
    print("Calling main function to process the PDF...hisadhiawdhiwa")
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
    
    process_multiple_pdfs(folder_path=pdf_path, save_answer=save_answer,save_questions=save_questions,
                          Qu_model=Qu_model,ID_model=id_model,device=device,predict_csv=predict_csv)
 

def run_main_process(Qu_model_path, ID_model_path, pdf_path, out_path, save_answer, save_questions, predict_csv, csv_path, save_result):
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
    process_multiple_pdfs(folder_path=pdf_path, save_answer=save_answer,save_questions=save_questions,
                          Qu_model=Qu_model,ID_model=id_model,device=device,predict_csv=predict_csv)
    compare_answers(csv_path, predict_csv, save_result)
    end_time_main = time.time()
    print(f"main function took {end_time_main - start_time_main} seconds to run.")






if __name__ == "__main__":
    Qu_model_path = 'Question_Model/4CN_bz8_lr0.0005_ep45_3'
    ID_model_path = 'ID_Model/ID_lr0.00005_ep30'
    # pdf_path = "PDF_Document/test_3.pdf"
    pdf_path = "PDF_Document/Multiple_PDF_test"
    out_path = "JPG_Document/TruthData"
    save_answer = 'Answer_area/test_new'
    save_questions = 'Validation/test_new'
    # save_ID = 'ID_middle' ID的中间值的保存


    correct_answer = 'results_txt/Correct_Answer.csv'
    predict_csv = 'results_txt/PDF_Answer.csv'
    result_csv = 'results_txt/Student_Scores.csv'
    feedback = 'results_txt/Feedback'

    # 记录 main 函数的开始时间
    main(Qu_model_path,ID_model_path, pdf_path, out_path, save_answer,save_questions,predict_csv)
    convert_csv_format(predict_csv, correct_answer)
    # compare_answers(correct_answer,predict_csv,result_csv)
    # run_main_process(Qu_model_path, ID_model_path,
    #                      pdf_path, out_path, save_answer, save_questions, predict_csv, correct_answer, result_csv)
    # student_to_txt(result_csv,feedback)





