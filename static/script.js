// 当 PDF 文件选择后的处理函数
function handlePDFUpload(event) {
    var files = event.target.files; // 获取选中的文件列表
    var pdfListContainer = document.getElementById('pdf-list');
    pdfListContainer.innerHTML = ''; // 先清空列表

    Array.from(files).forEach((file) => {
        if (confirm(`Do you want to add ${file.name}?`)) {
            // 为每个文件创建一个段落元素显示文件名
            var fileElement = document.createElement('div');
            fileElement.textContent = file.name;
            pdfListContainer.appendChild(fileElement);
        }
    });
}

// 当 CSV 文件选择后的处理函数
function handleCSVUpload(event) {
    var file = event.target.files[0]; // 获取选中的 CSV 文件
    var csvPreviewContainer = document.getElementById('csv-preview');
    csvPreviewContainer.innerHTML = ''; // 先清空预览

    if (file && confirm(`Do you want to add ${file.name}?`)) {
        var reader = new FileReader();
        reader.onload = function(e) {
            csvPreviewContainer.innerHTML = csvToTable(e.target.result); // 将 CSV 内容转换为表格并显示
        };
        reader.readAsText(file);
    }
}

// 辅助函数，将 CSV 文本转换为 HTML 表格
function csvToTable(csv) {
    var lines = csv.split('\n');
    var result = '<table>';
    
    lines.forEach(function(line) {
        result += '<tr>';
        var entries = line.split(',');
        entries.forEach(function(entry) {
            result += '<td>' + entry.trim() + '</td>';
        });
        result += '</tr>';
    });
    result += '</table>';
    return result;
}

// 监听文档加载事件以绑定函数
document.addEventListener('DOMContentLoaded', function() {
    var pdfUploadInput = document.getElementById('pdf-upload');
    var csvUploadInput = document.getElementById('csv-upload');
    var deletePdfBtn = document.getElementById('delete-pdf');
    var deleteCsvBtn = document.getElementById('delete-csv');
    
    // 绑定上传事件处理器
    pdfUploadInput.addEventListener('change', handlePDFUpload);
    csvUploadInput.addEventListener('change', handleCSVUpload);

    // 绑定删除PDFs按钮事件处理器
    deletePdfBtn.addEventListener('click', function() {
        if (confirm('Are you sure you want to delete all PDFs?')) {
            document.getElementById('pdf-list').innerHTML = '';
        }
    });

    // 绑定删除CSVs按钮事件处理器
    deleteCsvBtn.addEventListener('click', function() {
        if (confirm('Are you sure you want to delete the CSV preview?')) {
            document.getElementById('csv-preview').innerHTML = '';
        }
    });
});


































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

//     document.getElementById('experience-btn').addEventListener('click', function() {
//         // Start the loading animation
//         const loader = document.getElementById('loader');
//         loader.style.display = 'block';
    
//         fetch('/grade', { // Assuming '/grade' is the endpoint that triggers the grading process
//             method: 'POST',
//         })
//         .then(response => {
//             // Stop the loading animation when a response is received
//             document.getElementById('loader').style.display = 'none';
            
//             if(response.ok) {
//                 return response.blob(); // Assuming the response is the graded results in a CSV file
//             }
//             throw new Error('Network response was not ok.');
//         })
//         .then(blob => {
//             const url = window.URL.createObjectURL(blob);
//             const link = document.createElement('a');
//             link.href = url;
//             link.download = 'graded_results.csv'; // Assuming you want to download the results as a CSV
//             document.body.appendChild(link);
//             link.click();
//             document.body.removeChild(link);
//         })
//         .catch(error => {
//             console.error('Error:', error);
//         });
//     });
// });
