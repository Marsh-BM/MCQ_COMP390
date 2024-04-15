// 当 PDF 文件选择后的处理函数
function handlePDFUpload(event) {
    var files = event.target.files; // 获取选中的文件列表
    if (files.length === 0) return; // 如果没有文件被选中，直接返回



    // 弹出确认提交的对话框
    if (!confirm("Do you want to submit these PDF files?")) {
        return; // 如果用户选择取消，什么也不做
    }

    // 准备表单数据并上传
    var formData = new FormData();
    Array.from(files).forEach(file => {
        formData.append('pdf-file', file);
    });

    fetch('/upload_pdf', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('PDF Upload Success:', data);
        // 这里可以添加上传成功后的其他操作
        displayUploadedFiles();
    })
    .catch(error => {
        console.error('Error uploading PDF:', error);
    });

    // 显示文件名
    
}

// 当 CSV 文件选择后的处理函数
function handleCSVUpload(event) {
    var file = event.target.files[0]; // 获取选中的CSV文件
    if (!file) return; // 如果没有文件被选中，直接返回

    // 显示文件名
    // displayUploadedFiles([file], 'csv'); // 因为只有一个文件，所以用数组包装

    // 弹出确认提交的对话框
    if (!confirm("Do you want to submit this CSV file?")) {
        return; // 如果用户选择取消，什么也不做
    }

    // 准备表单数据并上传
    var formData = new FormData();
    formData.append('file', file);

    fetch('/upload_ans', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Answer Upload Success:', data);
        if(file.type === 'text/csv'){
            previewCSV(event); // 显示CSV内容预览
            document.getElementById('pdf-preview').style.display = 'none'; // 隐藏 PDF 预览
        }else if (file.type === 'application/pdf'){
            previewPDF(file)
            document.getElementById('csv-preview').style.display = 'none';
        }
        
    })
    .catch(error => {
        console.error('Error uploading CSV:', error);
    });
}

// Function to handle student CSV file upload
function handleStudentCSVUpload(event) {
    // console.log("handleStudentCSVUpload called");
    var file = event.target.files[0]; // Get the selected file
    if (!file) return; // If no file was selected, return immediately

    // Ask for confirmation before uploading the file
    if (!confirm("Do you want to submit this student information Students CSV file?")) {
        return; // If the user cancels, do nothing
    }

    // Prepare the form data for uploading
    var formData = new FormData();
    formData.append('student-csv-file', file); // Make sure the 'name' attribute in your HTML matches this

    // Use the fetch API to send the student CSV file to the server
    fetch('/upload_student_csv', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Student CSV Upload Success:', data);
        previewStudentsCSV(event)
    })
    .catch(error => {
        console.error('Error uploading student CSV:', error);
    });
}



// 用于显示上传的文件列表
function displayUploadedFiles() {
    var output = document.getElementById('pdf-file-names');

    // 发送请求到 Flask 服务器获取文件列表
    fetch('/files/pdf').then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
    }).then(files => {
        // 清空之前的内容
        output.innerHTML = '';

        // 显示文件列表
        files.forEach((file, index) => {
            var fileNameDisplay = document.createElement('p');
            fileNameDisplay.textContent = `File ${index + 1}: ${file}`;
            output.appendChild(fileNameDisplay);
        });

        // 显示输出
        output.style.display = 'block';
    }).catch(error => {
        console.error('Failed to fetch files:', error);
    });
}


// 显示CSV文件内容的预览
function previewCSV(event) {
    var file = event.target.files[0]; // 获取选中的CSV文件
    if (!file) return;

    var reader = new FileReader();
    reader.onload = function(e) {
        var csvContent = e.target.result;
        var previewContainer = document.getElementById('csv-preview');
        previewContainer.innerHTML = csvToTable(csvContent);
        previewContainer.style.display = 'block';
    };
    reader.readAsText(file);
}

// 显示PDF文件内容的预览
function previewPDF(file) {
    var url = URL.createObjectURL(file); // 创建一个指向该文件的 URL
    var pdfPreview = document.getElementById('pdf-preview');
    pdfPreview.src = url;
    pdfPreview.style.display = 'block'; // 显示预览
}

// 显示Students CSV文件内容的预览
function previewStudentsCSV(event) {
    var file = event.target.files[0]; // 获取选中的CSV文件
    if (!file) return;

    var reader = new FileReader();
    reader.onload = function(e) {
        var csvContent = e.target.result;
        var previewContainer = document.getElementById('student-csv-preview');
        previewContainer.innerHTML = csvToTable(csvContent);
        previewContainer.style.display = 'block';
    };
    reader.readAsText(file);
}

// 辅助函数，将CSV文本转换为HTML表格
function csvToTable(csv) {
    var lines = csv.split('\n');
    var html = '<table>';
    lines.forEach(function(line, index) {
        html += '<tr>';
        var cells = line.split(',');
        cells.forEach(function(cell) {
            html += `<td>${cell.trim()}</td>`;
        });
        html += '</tr>';
    });
    html += '</table>';
    return html;
}

function deletePDFFiles() {
    // if (!confirm('Are you sure you want to delete all PDF files?')) return;

    fetch('/delete_pdf_files', { method: 'POST' })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Delete PDF files success:', data);
        displayUploadedFiles('pdf');
    })
    .catch(error => {
        console.error('Error deleting PDF files:', error);
    });
}

function deleteAnswerFiles() {
    // if (!confirm('Are you sure you want to delete all CSV files?')) return;

    fetch('/delete_ans_files', { method: 'POST' })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Delete CSV files success:', data);
        document.getElementById('csv-preview').innerHTML = '';
        document.getElementById('pdf-preview').style.display = 'none';
        document.getElementById('csv-preview').style.display = 'block';
    })
    .catch(error => {
        console.error('Error deleting CSV files:', error);
    });
}

function deleteStudentCSV() {
    // if (!confirm('Are you sure you want to delete all CSV files?')) return;

    fetch('/delete_student_csv', { method: 'POST' })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Delete Students CSV files success:', data);
        document.getElementById('student-csv-preview').innerHTML = '';
    })
    .catch(error => {
        console.error('Error deleting Students CSV files:', error);
    });
}

// 初始化函数，用于设置事件监听器
function initialize() {
    var pdfFileInput = document.getElementById('pdf-file-input');
    if (pdfFileInput) {
        pdfFileInput.addEventListener('change', handlePDFUpload);
    }
    
    var csvFileInput = document.getElementById('ans-file-input');
    if (csvFileInput) {
        csvFileInput.addEventListener('change', handleCSVUpload);
    }

    var studentCsvFileInput = document.getElementById('student-csv-file-input'); 
    if (studentCsvFileInput) {
        studentCsvFileInput.addEventListener('change', handleStudentCSVUpload);
    }


    var pdfdelete = document.getElementById('delete-pdf-btn')
    if(pdfdelete){
        pdfdelete.addEventListener('click', function(event){
            if (confirm('Are you sure you want to delete all PDF files?')) {
                deletePDFFiles();
            }
        });
    }
    var csvdelete = document.getElementById('delete-csv-btn')
    if(csvdelete){
        csvdelete.addEventListener('click',function(event){
            if (confirm('Are you sure you want to delete all CSV files?')) {
                deleteAnswerFiles();
            }
        });
    }
    var deleteStudentCsvBtn = document.getElementById('delete-student-csv-btn');
    if (deleteStudentCsvBtn) {
        deleteStudentCsvBtn.addEventListener('click', function(event) {
            if (confirm('Are you sure you want to delete the student CSV file?')) {
                deleteStudentCSV();
            }
        });
    }
}

document.addEventListener('DOMContentLoaded', initialize);
document.getElementById('experience-btn').addEventListener('click', function() {
    const btn = this; // 保存按钮引用，以便在Promise链中使用

    // 检查文件夹
    fetch('/check_folders')
    .then(response => response.json())
    .then(data => {
        if (data.is_pdf_empty || data.is_csv_empty) {
            alert("PDF or CSV folders are empty. Please upload files before grading.");
            btn.style.display = 'block';
            return; // 终止后续操作
        }
       
        btn.style.display = 'none';
        // 显示加载动画
        const loader = document.getElementById('loader');
        loader.style.display = 'block';

        // 发送POST请求到后端的/grade路由
        return fetch('/grade', {
            method: 'POST',
        });
    })
    .then(response => {
        if (!response) return; // 如果之前终止了操作，这里直接返回
        
        loader.style.display = 'none'; // 响应接收后隐藏加载动画

        if(response.ok) {
            return response.blob(); // 假设响应是一个要下载的CSV文件
        }
        throw new Error('Network response was not ok.');
    })
    .then(blob => {
        if (!blob) return; // 如果之前终止了操作，这里直接返回
        
        // 创建URL并模拟<a>点击以下载文件
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'graded_results.csv';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        btn.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        alert("There was an error processing your request.");
        btn.style.display = 'block';
    });
});
