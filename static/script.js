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
    
    // 循环处理每一行
    lines.forEach(function(line) {
        result += '<tr>';
        var entries = line.split(',');
        
        // 循环处理每一列
        entries.forEach(function(entry) {
            result += '<td>' + entry.trim() + '</td>'; // 使用 trim() 去除空白字符
        });
        result += '</tr>';
    });
    result += '</table>';
    return result;
}

// 确保绑定 input 元素的 change 事件
document.addEventListener('DOMContentLoaded', function() {
    var pdfFileInput = document.getElementById('pdf-file-input');
    if (pdfFileInput) {
        pdfFileInput.addEventListener('change', previewPDF);
    }
    
    var csvFileInput = document.getElementById('csv-file-input');
    if (csvFileInput) {
        csvFileInput.addEventListener('change', previewCSV);
    }
});

// 点击“立即体验”按钮上传文件
document.getElementById('submit-btn').addEventListener('click', function() {
    console.log("已点击")
    var formData = new FormData(document.getElementById('file-upload-form'));
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (response.ok) {
            return response.json();
        } else {
            throw new Error('Server response was not OK');
        }
    })
    .then(data => {
        console.log(data);
        // 处理服务器响应的数据
    })
    .catch(error => {
        console.error('Error:', error);
    });
});