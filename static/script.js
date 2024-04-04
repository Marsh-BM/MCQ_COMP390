// 当 PDF 文件选择后的处理函数
function previewPDF(event) {
    var output = document.getElementById('pdf-preview');
    output.src = URL.createObjectURL(event.target.files[0]);
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
        var formData = new FormData(document.getElementById('file-upload-form'));
        // 仅保留PDF文件
        formData.delete('csv-file'); // 移除CSV文件
        fetch('/upload', {
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
        var formData = new FormData(document.getElementById('file-upload-form'));
        // 仅保留CSV文件
        formData.delete('pdf-file'); // 移除PDF文件
        fetch('/upload', {
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
// document.getElementById('experience-btn').addEventListener('click', function() {
//     fetch('/grade', { // Assuming '/grade' is the endpoint that triggers the grading process
//         method: 'POST',
//     })
//     .then(response => {
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

document.getElementById('experience-btn').addEventListener('click', function() {
    fetch('/grade', { // Assuming '/grade' is the endpoint that triggers the grading process
        method: 'POST',
    })
    .then(response => {
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
