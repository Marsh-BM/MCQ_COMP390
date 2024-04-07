// 当 PDF 文件选择后的处理函数
function previewPDF(event) {
    var output = document.getElementById('pdf-preview');
    output.innerHTML = ''; // 清空之前的内容
    Array.from(event.target.files).forEach((file, index) => {
        var fileNameDisplay = document.createElement('p');
        fileNameDisplay.textContent = `File ${index + 1}: ${file.name}`;
        output.appendChild(fileNameDisplay);
    });
    output.style.display = 'block'; // 显示预览
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
