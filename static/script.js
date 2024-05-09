// Handling function when a PDF file is selected
function handlePDFUpload(event) {
    var files = event.target.files; 
    if (files.length === 0) return; 



    // Pop-up dialog to confirm submission
    if (!confirm("Do you want to submit these PDF files?")) {
        return; 
    }

    // Prepare form data and upload
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
        displayUploadedFiles();
    })
    .catch(error => {
        console.error('Error uploading PDF:', error);
    });
}

// Handling function when CSV file is selected
function handleCSVUpload(event) {
    var file = event.target.files[0]; 
    if (!file) return; 

    // Pop-up dialog to confirm submission
    if (!confirm("Do you want to submit this CSV file?")) {
        return; 
    }

    // Prepare form data and upload
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
            previewCSV(event); 
            document.getElementById('pdf-preview').style.display = 'none'; 
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



// Used to display a list of uploaded files
function displayUploadedFiles() {
    var output = document.getElementById('pdf-file-names');

    // Send a request to the Flask server for a list of files.
    fetch('/files/pdf').then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
    }).then(files => {
        // Clear the previous content
        output.innerHTML = '';

        // Display the list of files
        files.forEach((file, index) => {
            var fileNameDisplay = document.createElement('p');
            fileNameDisplay.textContent = `File ${index + 1}: ${file}`;
            output.appendChild(fileNameDisplay);
        });

        // Display output
        output.style.display = 'block';
    }).catch(error => {
        console.error('Failed to fetch files:', error);
    });
}


// Displays a preview of the contents of the CSV file
function previewCSV(event) {
    var file = event.target.files[0];
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

// Display a preview of the contents of the PDF file
function previewPDF(file) {
    var url = URL.createObjectURL(file); // Create a URL to the file
    var pdfPreview = document.getElementById('pdf-preview');
    pdfPreview.src = url;
    pdfPreview.style.display = 'block'; // Show preview
}

// Show a preview of the contents of the Students CSV file.
function previewStudentsCSV(event) {
    var file = event.target.files[0]; // Get the selected CSV file
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

// helper function to convert CSV text to HTML tables
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

// Initialization function to set up event listeners.
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
    const btn = this; 
    // Check if the PDF and CSV folders are empty
    fetch('/check_folders')
    .then(response => response.json())
    .then(data => {
        if (data.is_pdf_empty || data.is_csv_empty) {
            alert("PDF or CSV folders are empty. Please upload files before grading.");
            btn.style.display = 'block';
            return; 
        }
        // Hide the button
        btn.style.display = 'none';
        // Display loader
        const loader = document.getElementById('loader');
        loader.style.display = 'block';

        // Send a request to the server to grade the PDF files
        return fetch('/grade', {
            method: 'POST',
        });
    })
    .then(response => {
        if (!response) return; 
        
        loader.style.display = 'none'; 

        if(response.ok) {
            return response.blob(); 
        }
        throw new Error('Network response was not ok.');
    })
    .then(blob => {
        if (!blob) return; 
        
        // Download the graded results
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
