// // 当 PDF 文件选择后的处理函数
// function previewPDF(event) {
//     var output = document.getElementById('pdf-preview');
//     output.src = URL.createObjectURL(event.target.files[0]);
//     output.style.display = 'block'; // 显示预览
// }
// function previewPDF(event) {
//     var files = event.target.files; // 获取选中的文件列表
//     if (files.length > 0) {
//         // 弹出确认提交的对话框
//         var confirmSubmission = confirm("Do you want to submit these files?");
//         if (confirmSubmission) {
//             // 用户确认提交，显示文件名和编号
//             var output = document.getElementById('pdf-file-names');
//             output.innerHTML = ''; // 清空之前的内容

//             Array.from(files).forEach((file, index) => {
//                 // 为每个文件创建一个段落元素
//                 var fileNameDisplay = document.createElement('p');
//                 fileNameDisplay.textContent = `File ${index + 1}: ${file.name}`;
//                 output.appendChild(fileNameDisplay); // 将段落元素添加到显示容器中
//             });

//             // 确保显示容器是可见的
//             output.style.display = 'block'; 
//         } else {
//             // 用户取消提交，可以选择重置文件输入或不做任何操作
//             // 为了简化，这里我们选择不做任何操作
//         }
//     }
// }


// // 当 CSV 文件选择后的处理函数
// function previewCSV(event) {
//     var reader = new FileReader();
//     reader.onload = function() {
//         var csvOutput = document.getElementById('csv-preview');
//         csvOutput.innerHTML = '<table>' + csvToTable(reader.result) + '</table>';
//         csvOutput.style.display = 'block'; // 显示预览
//     };
//     reader.readAsText(event.target.files[0]);
// }

// // 辅助函数，将 CSV 文本转换为 HTML 表格
// function csvToTable(csv) {
//     var lines = csv.split('\n');
//     var result = '<table>';
    
//     lines.forEach(function(line) {
//         result += '<tr>';
//         var entries = line.split(',');
//         entries.forEach(function(entry) {
//             result += '<td>' + entry.trim() + '</td>';
//         });
//         result += '</tr>';
//     });
//     result += '</table>';
//     return result;
// }

// document.addEventListener('DOMContentLoaded', function() {
//     var pdfFileInput = document.getElementById('pdf-file-input');
//     if (pdfFileInput) {
//         pdfFileInput.addEventListener('change', previewPDF);
//     }
    
//     var csvFileInput = document.getElementById('csv-file-input');
//     if (csvFileInput) {
//         csvFileInput.addEventListener('change', previewCSV);
//     }

//     // 为更新PDF按钮添加上传功能
//     document.getElementById('update-pdf').addEventListener('click', function() {
//         var formData = new FormData(document.getElementById('file-upload-pdf'));
//         fetch('/upload_pdf', { // Ensure this matches the Flask route exactly
//             method: 'POST',
//             body: formData
//         })
//         .then(response => response.json())
//         .then(data => {
//             console.log(data);
//             // 上传成功后的操作
//         })
//         .catch(error => {
//             console.error('Error:', error);
//         });
//     });

//     // 为更新CSV按钮添加上传功能
//     document.getElementById('update-csv').addEventListener('click', function() {
//         var formData = new FormData(document.getElementById('file-upload-csv'));
//         fetch('/upload_csv', { // Ensure this matches the Flask route exactly
//             method: 'POST',
//             body: formData
//         })
//         .then(response => response.json())
//         .then(data => {
//             console.log(data);
//             // 上传成功后的操作
//         })
//         .catch(error => {
//             console.error('Error:', error);
//         });
//     });

    // document.getElementById('experience-btn').addEventListener('click', function() {
    //     // Start the loading animation
    //     const loader = document.getElementById('loader');
    //     loader.style.display = 'block';
    
    //     fetch('/grade', { // Assuming '/grade' is the endpoint that triggers the grading process
    //         method: 'POST',
    //     })
    //     .then(response => {
    //         // Stop the loading animation when a response is received
    //         document.getElementById('loader').style.display = 'none';
            
    //         if(response.ok) {
    //             return response.blob(); // Assuming the response is the graded results in a CSV file
    //         }
    //         throw new Error('Network response was not ok.');
    //     })
    //     .then(blob => {
    //         const url = window.URL.createObjectURL(blob);
    //         const link = document.createElement('a');
    //         link.href = url;
    //         link.download = 'graded_results.csv'; // Assuming you want to download the results as a CSV
    //         document.body.appendChild(link);
    //         link.click();
    //         document.body.removeChild(link);
    //     })
    //     .catch(error => {
    //         console.error('Error:', error);
    //     });
    // });
// });

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
        displayUploadedFiles();
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
    formData.append('csv-file', file);

    fetch('/upload_csv', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('CSV Upload Success:', data);
        previewCSV(event); // 显示CSV内容预览
    })
    .catch(error => {
        console.error('Error uploading CSV:', error);
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

function deleteCSVFiles() {
    // if (!confirm('Are you sure you want to delete all CSV files?')) return;

    fetch('/delete_csv_files', { method: 'POST' })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Delete CSV files success:', data);
        document.getElementById('csv-preview').innerHTML = '';
    })
    .catch(error => {
        console.error('Error deleting CSV files:', error);
    });
}



// 初始化函数，用于设置事件监听器
function initialize() {
    var pdfFileInput = document.getElementById('pdf-file-input');
    if (pdfFileInput) {
        pdfFileInput.addEventListener('change', handlePDFUpload);
    }
    
    var csvFileInput = document.getElementById('csv-file-input');
    if (csvFileInput) {
        csvFileInput.addEventListener('change', handleCSVUpload);
    }

    var csvFileInput = document.getElementById('csv-file-input');
    if (csvFileInput) {
        csvFileInput.addEventListener('change', function(event) {
            // 这里假设用户已经通过确认对话框同意上传
            previewCSV(event); // 显示CSV内容预览
        });
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
                deleteCSVFiles();
            }
        })
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
