document.addEventListener('DOMContentLoaded', function() {
    var pdfUploadBtn = document.getElementById('pdf-upload');
    var csvUploadBtn = document.getElementById('csv-upload');
    var deletePdfBtn = document.getElementById('delete-pdf');
    var deleteCsvBtn = document.getElementById('delete-csv');

    pdfUploadBtn.addEventListener('change', function(event) {
        if (confirm('Do you want to add the selected PDFs?')) {
            var fileList = event.target.files;
            var pdfListContainer = document.getElementById('pdf-list');
            pdfListContainer.innerHTML = ''; // 清空现有的列表
            for (var i = 0; i < fileList.length; i++) {
                var file = fileList[i];
                var fileElement = document.createElement('div');
                fileElement.textContent = file.name;
                pdfListContainer.appendChild(fileElement);
            }
        }
    });

    csvUploadBtn.addEventListener('change', function(event) {
        if (confirm('Do you want to add the selected CSV?')) {
            var file = event.target.files[0];
            var reader = new FileReader();
            reader.onload = function(e) {
                var csvPreviewContainer = document.getElementById('csv-preview');
                csvPreviewContainer.innerHTML = e.target.result; // 显示CSV内容
            };
            reader.readAsText(file);
        }
    });

    deletePdfBtn.addEventListener('click', function() {
        if (confirm('Are you sure you want to delete all PDFs?')) {
            var pdfListContainer = document.getElementById('pdf-list');
            pdfListContainer.innerHTML = ''; // 清空PDF列表
        }
    });

    deleteCsvBtn.addEventListener('click', function() {
        if (confirm('Are you sure you want to delete the CSV preview?')) {
            var csvPreviewContainer = document.getElementById('csv-preview');
            csvPreviewContainer.innerHTML = ''; // 清空CSV预览
        }
    });
});
