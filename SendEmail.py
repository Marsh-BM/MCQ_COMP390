import os
import csv
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
                
        
                
                answers = row['Answer'].split()  # Splitting the answers by space
                wrong_answers = [ans for ans in answers if any(char.islower() or char == '-' for char in ans)]
                # Format the wrong answers with their question numbers
                wrong_answers_formatted = [f"{ans}" for i, ans in enumerate(wrong_answers)]
                wrong_questions = ', '.join(wrong_answers_formatted) if wrong_answers else 'None'
                student_file.write(f"Wrong Questions: {wrong_questions}\n")

def read_students_info(csv_file_path):

    students_info = []
    with open(csv_file_path, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            students_info.append((row['ID'], row['Name'], row['Email']))
    return students_info

def send_email_with_attachment(smtp_user, smtp_password, student_email, student_name, attachment_path):

    # smtp_adress = ''
    # smtp_password=''
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    
    message = MIMEMultipart()
    message['From'] = smtp_user
    message['To'] = student_email
    message['Subject'] = "Test grade report"
    
    body = f"Dear {student_name}，Your test score report is attached to the email."
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

def send_reports_to_students(grades_csv_path,students_info_csv_path, txt_files_dir, smtp_user, smtp_password):

    students_info = read_students_info(students_info_csv_path)
    for student_id, student_name, student_email in students_info:
        attachment_path = os.path.join(txt_files_dir, f"{student_id}.txt")
        if os.path.exists(attachment_path):
            send_email_with_attachment(smtp_user, smtp_password, student_email, student_name, attachment_path)
        else:
            print(f"Warning: Cannot find the ID:{student_id} scores report。")





if __name__ == "__main__":
    grades_csv_path = 'results_txt\Student_Scores.csv'
    students_info_csv_path = 'results_txt\Information of students.csv' 
    output_dir = 'results_txt\Feedback' 
    smtp_user = 'marshbm0518@gmail.com' 
    smtp_password = 'xbok hbdr fhxp hkyh'

    student_to_txt(grades_csv_path, output_dir)

    send_reports_to_students(grades_csv_path, students_info_csv_path, output_dir, smtp_user, smtp_password)
