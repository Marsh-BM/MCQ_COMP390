from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
from main import run_main_process, main,convert_csv_format
from SendEmail import student_to_txt, send_reports_to_students
import shutil
from flask import jsonify

app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config['UPLOAD_FOLDER_PDF'] = 'uploaded_PDF'
app.config['UPLOAD_FOLDER_ANSWER'] = 'uploaded_Answer'
app.config['UPLOAD_FOLDER_STUDENTS'] = 'uploaded_Students'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'csv'}
app.config['RESULT_FOLDER'] = 'results_txt'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER_PDF'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER_ANSWER'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER_STUDENTS'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)  # Ensure the result folder exists

uploaded_files = {
    'pdf': [],
    'ans': None,
    'student_csv': None  # Add this line to track the student info CSV
}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/files/pdf')
def files_pdf():
    pdf_folder = app.config['UPLOAD_FOLDER_PDF']
    try:
        pdf_files = os.listdir(pdf_folder)
        pdf_files = [f for f in pdf_files if f.lower().endswith('.pdf')]
        return jsonify(pdf_files)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/delete_pdf_files', methods=['POST'])
def delete_pdf_files():
    try:
        pdf_folder = app.config['UPLOAD_FOLDER_PDF']
        for filename in os.listdir(pdf_folder):
            if filename.lower().endswith('.pdf'):
                os.unlink(os.path.join(pdf_folder, filename))
        return jsonify({'message': 'All PDF files have been deleted.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_ans_files', methods=['POST'])
def delete_ans_files():
    try:
        ans_folder = app.config['UPLOAD_FOLDER_ANSWER']
        for filename in os.listdir(ans_folder):
            file_path = os.path.join(ans_folder, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        return jsonify({'message': 'All files have been deleted.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/delete_student_csv', methods=['POST'])
def delete_student_csv():
    try:
        csv_folder = app.config['UPLOAD_FOLDER_STUDENTS']
        for filename in os.listdir(csv_folder):
            if filename.lower().endswith('.csv'):
                os.unlink(os.path.join(csv_folder, filename))
        return jsonify({'message': 'All CSV files have been deleted.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_folders')
def check_folders():
    pdf_folder = app.config['UPLOAD_FOLDER_PDF']
    ans_folder = app.config['UPLOAD_FOLDER_ANSWER']
    

    is_pdf_empty = not os.listdir(pdf_folder)
    is_ans_empty = not os.listdir(ans_folder)
    

    return jsonify({
        'is_pdf_empty': is_pdf_empty,
        'is_ans_empty': is_ans_empty
    })


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    files = request.files.getlist('pdf-file') 
    if not files:
        return jsonify({'error': 'No file selected'}), 400

    uploaded_files['pdf'].clear()  

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER_PDF'], filename)
            file.save(file_path)
            uploaded_files['pdf'].append(file_path)  
        else:
            return jsonify({'error': 'File type not allowed'}), 400

    return jsonify({'message': 'PDF uploaded successfully', 'files': uploaded_files['pdf']})



@app.route('/upload_ans', methods=['POST'])
def upload_ans():
    file = request.files.get('file') 
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()

        if file_extension == 'csv':
            
            ans_csv_path = os.path.join(app.config['UPLOAD_FOLDER_ANSWER'], filename)
            file.save(ans_csv_path)
            uploaded_files['ans'] = filename  
            return jsonify({'message': 'CSV uploaded successfully.', 'filename': filename})
        elif file_extension == 'pdf':
            
            answer_pdf_path = os.path.join(app.config['UPLOAD_FOLDER_ANSWER'], filename)
            file.save(answer_pdf_path)

            
            try:
                print("Start")
                Qu_model_path = 'Question_Model/4CN_bz8_lr0.0005_ep45_3'
                ID_model_path = 'ID_Model/ID_lr0.00005_ep30'
                ans_pdf_path = os.path.join(app.config['UPLOAD_FOLDER_ANSWER'])
                print(ans_pdf_path)
                out_path = "JPG_Document/TruthData"
                save_answer = 'Answer_area/test_new'
                save_questions = 'Validation/test_new'
                predict_csv = 'uploaded_Middle/PDF_Answer.csv'
                csv_file_path = 'Correct_Answer.csv'

                main(Qu_model_path,ID_model_path, ans_pdf_path, out_path, save_answer,save_questions,predict_csv)
                convert_csv_format(predict_csv, os.path.join(app.config['UPLOAD_FOLDER_ANSWER'], 'Correct_Answer.csv'))
                print("End")
                # 
                # os.remove(answer_pdf_path)
                uploaded_files['ans'] = csv_file_path

                return jsonify({'message': 'CSV created from PDF successfully.', 'filename': predict_csv})
            except Exception as e:
                return jsonify({'error': 'Failed to convert PDF to CSV: ' + str(e)}), 500
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    else:
        return jsonify({'error': 'Invalid file type'}), 400



@app.route('/upload_student_csv', methods=['POST'])
def upload_student_csv():
    if 'student-csv-file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['student-csv-file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER_STUDENTS'], filename)
        file.save(file_path)
        uploaded_files['student_csv'] = filename  # Update the file name in the dictionary
        return jsonify({'message': 'Student CSV uploaded successfully.', 'file_path': file_path})
    else:
        return jsonify({'error': 'Invalid file type.'}), 400



@app.route('/grade', methods=['POST'])
def grade():
    print('Run grading process.')
    print(uploaded_files['pdf'])
    print(uploaded_files['ans'])
    if uploaded_files['pdf'] and uploaded_files['ans']:

        pdf_paths = 'uploaded_PDF'
        csv_path = os.path.join(app.config['UPLOAD_FOLDER_ANSWER'], uploaded_files['ans'])
        print(csv_path)
        out_path = "JPG_Document/TruthData"
        save_answer = 'Answer_area/test_new'
        save_questions = 'Validation/test_new'
        predict_csv = 'results_txt/ID_Question.csv' 

        save_result = os.path.join(app.config['RESULT_FOLDER'], 'Student_Scores.csv')
        Question_model = 'Question_Model/4CN_bz8_lr0.0005_ep45_3'
        ID_model = 'ID_Model/ID_lr0.00005_ep30'

        run_main_process(Question_model, ID_model,
                         pdf_paths, out_path, save_answer, save_questions, predict_csv, csv_path, save_result)
        

        feedback = 'Feedback'
        students_path = os.path.join(app.config['UPLOAD_FOLDER_STUDENTS'], uploaded_files['student_csv'])
        smtp_user = 'marshbm0518@gmail.com' # 你的Gmail邮箱地址
        smtp_password = 'xbok hbdr fhxp hkyh'
        student_to_txt(save_result, feedback)
        send_reports_to_students(save_result, students_path, feedback, smtp_user,smtp_password)


        
        # return jsonify({'message': 'Grading process completed successfully.', 'result_path': save_result})
        response = send_from_directory(app.config['RESULT_FOLDER'], 'Student_Scores.csv', as_attachment=True)
        
        # Delete the uploaded files after grading
        try:
            pdf_folder = app.config['UPLOAD_FOLDER_PDF']
            csv_folder = app.config['UPLOAD_FOLDER_ANSWER']
            student_folder = app.config['UPLOAD_FOLDER_STUDENTS']
            feedback = 'Feedback'
            # Delete the uploaded files
            for filename in os.listdir(pdf_folder):
                file_path = os.path.join(pdf_folder, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            
            # Delete the uploaded files
            for filename in os.listdir(csv_folder):
                file_path = os.path.join(csv_folder, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            # Delete the uploaded files
            for filename in os.listdir(student_folder):
                file_path = os.path.join(student_folder, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            # Delete the uploaded files
            for filename in os.listdir(feedback):
                file_path = os.path.join(feedback, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            print("Uploaded files have been deleted.")
        except Exception as e:
            print(f"Error deleting uploaded files: {e}")

        return response
    
    return "No files uploaded or one of the files is missing."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

