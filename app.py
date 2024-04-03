from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded_files'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'csv'}

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 检查文件扩展名是否允许
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def PageOne():
    return render_template('Home.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    # 检查是否有文件在请求中
    if 'pdf-file' not in request.files or 'csv-file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    pdf_file = request.files['pdf-file']
    csv_file = request.files['csv-file']
    
    # 如果用户没有选择文件，浏览器可能会提交一个空的文件
    if pdf_file.filename == '' or csv_file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if pdf_file and allowed_file(pdf_file.filename):
        pdf_filename = secure_filename(pdf_file.filename)
        pdf_file.save(os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename))
        
    if csv_file and allowed_file(csv_file.filename):
        csv_filename = secure_filename(csv_file.filename)
        csv_file.save(os.path.join(app.config['UPLOAD_FOLDER'], csv_filename))

    # 上传成功后的逻辑，这里简单地重定向回首页
    return redirect(url_for('PageOne'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

