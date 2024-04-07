// 当 PDF 文件选择后的处理函数
// function previewPDF(event) {
//     var output = document.getElementById('pdf-preview');
//     output.src = URL.createObjectURL(event.target.files[0]);
//     output.style.display = 'block'; // 显示预览
// }
function previewPDF(event) {
    // 获取用于显示文件名的容器
    var output = document.getElementById('pdf-file-names');
    output.innerHTML = ''; // 清空之前的内容

    // 遍历每个选中的文件
    Array.from(event.target.files).forEach((file, index) => {
        // 为每个文件创建一个段落元素
        var fileNameDisplay = document.createElement('p');
        fileNameDisplay.textContent = `File ${index + 1}: ${file.name}`;
        output.appendChild(fileNameDisplay); // 将段落元素添加到显示容器中
    });

    // 确保显示容器是可见的（如果之前有设置特定的样式来隐藏它的话）
    output.style.display = 'block'; 
}

// 当 CSV 文件选择后的处理函数
function previewCSV(event) {
    var reader = new FileReader();
    reader.onload = function() {
        var csvOutput = document.getElementById('csv-preview');
        csvOutput.innerHTML = '<table>' + csvToTable(reader.result) + '</table>';
        csvOutput.style.display = 'block'; // 显示预览
    };
    reader.readAsText(event.target.files[0]);
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

document.addEventListener('DOMContentLoaded', function() {
    var pdfFileInput = document.getElementById('pdf-file-input');
    if (pdfFileInput) {
        pdfFileInput.addEventListener('change', previewPDF);
    }
    
    var csvFileInput = document.getElementById('csv-file-input');
    if (csvFileInput) {
        csvFileInput.addEventListener('change', previewCSV);
    }

    // 为更新PDF按钮添加上传功能
    document.getElementById('update-pdf').addEventListener('click', function() {
        var formData = new FormData(document.getElementById('file-upload-pdf'));
        fetch('/upload_pdf', { // Ensure this matches the Flask route exactly
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            // 上传成功后的操作
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    // 为更新CSV按钮添加上传功能
    document.getElementById('update-csv').addEventListener('click', function() {
        var formData = new FormData(document.getElementById('file-upload-csv'));
        fetch('/upload_csv', { // Ensure this matches the Flask route exactly
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            // 上传成功后的操作
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    document.getElementById('experience-btn').addEventListener('click', function() {
        // Start the loading animation
        const loader = document.getElementById('loader');
        loader.style.display = 'block';
    
        fetch('/grade', { // Assuming '/grade' is the endpoint that triggers the grading process
            method: 'POST',
        })
        .then(response => {
            // Stop the loading animation when a response is received
            document.getElementById('loader').style.display = 'none';
            
            if(response.ok) {
                return response.blob(); // Assuming the response is the graded results in a CSV file
            }
            throw new Error('Network response was not ok.');
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'graded_results.csv'; // Assuming you want to download the results as a CSV
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        })
        .catch(error => {
            console.error('Error:', error);
        });

        
    });

});
