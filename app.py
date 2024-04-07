from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
from main import run_main_process  # 确保 main.py 在 Flask 应用的搜索路径中

app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config['UPLOAD_FOLDER_PDF'] = 'uploaded_PDF'
app.config['UPLOAD_FOLDER_CSV'] = 'uploaded_CSV'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'csv'}
app.config['RESULT_FOLDER'] = 'results_txt'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER_PDF'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER_CSV'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)  # Ensure the result folder exists

# 存储最近上传的PDF和CSV文件名
uploaded_files = {
    'pdf': None,
    'csv': None
}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    uploaded = False  # Flag to check if any file was uploaded

    if 'pdf-file' in request.files:
        pdf_file = request.files['pdf-file']
        if pdf_file and allowed_file(pdf_file.filename):
            pdf_filename = secure_filename(pdf_file.filename)
            pdf_file.save(os.path.join(app.config['UPLOAD_FOLDER_PDF'], pdf_filename))
            uploaded_files['pdf'] = pdf_filename
            uploaded = True

    if not uploaded:
        flash('No file selected or file type not allowed.')
        return redirect(request.url)

    flash('PDF uploaded successfully.')
    return redirect(url_for('home'))

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    uploaded = False  # Flag to check if any file was uploaded

    if 'csv-file' in request.files:
        csv_file = request.files['csv-file']
        if csv_file and allowed_file(csv_file.filename):
            csv_filename = secure_filename(csv_file.filename)
            csv_file.save(os.path.join(app.config['UPLOAD_FOLDER_CSV'], csv_filename))
            uploaded_files['csv'] = csv_filename
            uploaded = True

    if not uploaded:
        flash('No file selected or file type not allowed.')
        return redirect(request.url)
    
    flash('CSV uploaded successfully.')
    return redirect(url_for('home'))

@app.route('/grade', methods=['POST'])
def grade():
    if uploaded_files['pdf'] and uploaded_files['csv']:
        # 构建完整路径
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER_PDF'], uploaded_files['pdf'])
        csv_path = os.path.join(app.config['UPLOAD_FOLDER_CSV'], uploaded_files['csv'])
        out_path = "JPG_Document/TruthData"
        save_answer = 'Answer_area/test_new'
        save_questions = 'Validation/test_new'
        save_ID = 'ID_middle'
        # 这个csv是中间值，学生的答案
        save_csv = 'results_txt/ID_Question.csv' 
        # 这个是最终与正确答案比较后的结果，也就是我要返回给用户的csv
        save_result = os.path.join(app.config['RESULT_FOLDER'], 'Student_Scores.csv')
        # Question_model = 'Question_Model/lr0.0005_ep10'
        # Question_model = 'lr0.0005_ep20_###'
        # Question_model = 'lr0.0005_ep10_###'
        # Question_model = 'lr0.0005_ep20_###_new'
        # Question_model = 'bz8_lr0.0005_ep40_1'
        # Question_model = 'bz8_lr0.0005_ep40_2'
        # Question_model = 'bz8_lr0.0005_ep25_2'
        Question_model = 'bz8_lr0.0005_ep45_3'
        ID_model = 'ID_Model/ID_lr0.00005_ep30'

        run_main_process(Question_model, ID_model,
                         pdf_path, out_path, save_answer, save_questions, save_ID, save_csv, csv_path, save_result)
        
        # 下载结果
        response = send_from_directory(app.config['RESULT_FOLDER'], 'Student_Scores.csv', as_attachment=True)
        
        # 删除提供的PDF和CSV文件
        try:
            os.remove(pdf_path)
            os.remove(csv_path)
            print("Uploaded files have been deleted.")
        except Exception as e:
            print(f"Error deleting uploaded files: {e}")

        return response
    
    return "No files uploaded or one of the files is missing."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

