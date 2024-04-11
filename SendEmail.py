from pdf_to_image import pdf_to_png, process_images,process_multiple_pdfs
import os
import cv2
import torch
from Questions_model import Questions_model
from ID_model import ID_model
import csv
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


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
                
        
                # 将答案字符串拆分为独立的答案项
                answers = row['Answer'].split()

                answers = row['Answer'].split()  # Splitting the answers by space
                wrong_answers = [ans for ans in answers if any(char.islower() for char in ans)]
                # Format the wrong answers with their question numbers
                wrong_answers_formatted = [f"{ans}" for i, ans in enumerate(wrong_answers)]
                wrong_questions = ', '.join(wrong_answers_formatted) if wrong_answers else 'None'
                student_file.write(f"Wrong Questions: {wrong_questions}\n")

def read_students_info(csv_file_path):
    """
    读取包含学生学号、姓名和邮箱的CSV文件。
    """
    students_info = []
    with open(csv_file_path, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # 现在也包括了学生的姓名
            students_info.append((row['ID'], row['Name'], row['Email']))
    return students_info

def send_email_with_attachment(smtp_user, smtp_password, student_email, student_name, attachment_path):
    """
    发送带有成绩报告附件的邮件。
    """
    smtp_server = "smtp.gmail.com"
    
    smtp_port = 587
    
    message = MIMEMultipart()
    message['From'] = smtp_user
    message['To'] = student_email
    message['Subject'] = "考试成绩报告"
    
    body = f"亲爱的 {student_name}，您的考试成绩报告已附在邮件中。"
    message.attach(MIMEText(body, 'plain'))
    
    with open(attachment_path, 'rb') as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(attachment_path)}')
        message.attach(part)
    
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(message)

def send_reports_to_students(students_info_csv_path, txt_files_dir, smtp_user, smtp_password):
    """
    对CSV文件中的每个学生发送其成绩报告，包括学生的姓名。
    根据提供的学生信息CSV文件获取学生的邮箱和姓名。
    """
    students_info = read_students_info(students_info_csv_path)
    for student_id, student_name, student_email in students_info:
        attachment_path = os.path.join(txt_files_dir, f"{student_id}.txt")
        if os.path.exists(attachment_path):
            send_email_with_attachment(smtp_user, smtp_password, student_email, student_name, attachment_path)
        else:
            print(f"警告：找不到学号为 {student_id} 的学生的成绩报告文件。")



smtp_password = 'xbok hbdr fhxp hkyh'