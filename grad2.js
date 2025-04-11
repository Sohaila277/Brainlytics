document.addEventListener('DOMContentLoaded', function() {
    // Page navigation
    const navLinks = document.querySelectorAll('.nav-link');
    const pages = document.querySelectorAll('.page');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const pageId = this.getAttribute('data-page');
            
            // Hide all pages
            pages.forEach(page => {
                page.classList.remove('active');
            });
            
            // Show selected page
            document.getElementById(pageId).classList.add('active');
            
            // Update active nav link
            navLinks.forEach(navLink => {
                navLink.classList.remove('active');
            });
            this.classList.add('active');
        });
    });
    
    // Classifier page functionality
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('image-preview');
    const resultDiv = document.getElementById('result');
    const resultText = document.getElementById('result-text');
    const confidenceText = document.getElementById('confidence');
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('highlight');
    }
    
    function unhighlight() {
        dropArea.classList.remove('highlight');
    }
    
    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }
    
    // Handle selected files
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });
    
    function handleFiles(files) {
        if (files.length) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                    imagePreview.classList.remove('hidden');
                    resultDiv.classList.add('hidden');
                    
                    // Here you would typically send the image to your model for classification
                    // For now, we'll simulate a response
                    setTimeout(() => {
                        classifyImage(e.target.result);
                    }, 1500);
                };
                
                reader.readAsDataURL(file);
            } else {
                alert('Please upload an image file.');
            }
        }
    }
    
    function classifyImage(imageData) {
        // Show loading state
        resultText.textContent = "Processing...";
        confidenceText.textContent = "";
        resultDiv.classList.remove('hidden');
    
        // Convert data URL to Blob
        const blob = dataURLtoBlob(imageData);
        const formData = new FormData();
        formData.append('image', blob, 'upload.jpg');
    
        // Send to our Flask API
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            resultText.textContent = data.prediction;
            confidenceText.textContent = `Confidence: ${data.confidence.toFixed(2)}%`;
        })
        .catch(error => {
            console.error('Error:', error);
            resultText.textContent = 'Error: ' + error.message;
            confidenceText.textContent = '';
        });
    }
    
    // Helper function to convert data URL to Blob
    function dataURLtoBlob(dataURL) {
        const arr = dataURL.split(',');
        const mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n);
        while (n--) {
            u8arr[n] = bstr.charCodeAt(n);
        }
        return new Blob([u8arr], { type: mime });
    }
    
    // Initialize the home page as active
    document.querySelector('.nav-link[data-page="home"]').classList.add('active');
});