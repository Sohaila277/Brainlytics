document.addEventListener('DOMContentLoaded', function () {
    // Navigation
    const navLinks = document.querySelectorAll('.nav-link');
    const pages = document.querySelectorAll('.page');

    navLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const pageId = this.getAttribute('data-page');
            pages.forEach(page => page.classList.remove('active'));
            document.getElementById(pageId).classList.add('active');
        });
    });

    // Classification Tab
    const classifyDropArea = document.getElementById('drop-area');
    const classifyFileInput = document.getElementById('fileInput');
    const classifyPreview = document.getElementById('image-preview');
    const classifyResult = document.getElementById('result');

    setupFileUpload(classifyDropArea, classifyFileInput, classifyPreview, classifyResult, 'predict');

    // Segmentation Tab
    const segDropArea = document.getElementById('seg-drop-area');
    const segFileInput = document.getElementById('segFileInput');
    const segOriginal = document.getElementById('seg-original');
    const segMask = document.getElementById('seg-mask');
    const segOverlay = document.getElementById('seg-overlay');
    const segResult = document.getElementById('seg-result');

    setupFileUpload(segDropArea, segFileInput, segOriginal, segResult, 'segment', segMask, segOverlay);

    function setupFileUpload(dropArea, fileInput, previewElement, resultElement, endpoint, maskElement = null, overlayElement = null) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });

        dropArea.addEventListener('drop', handleDrop, false);
        fileInput.addEventListener('change', handleFileSelect);

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        function highlight() {
            dropArea.classList.add('highlight');
        }

        function unhighlight() {
            dropArea.classList.remove('highlight');
        }

        function handleDrop(e) {
            const files = e.dataTransfer.files;
            if (files.length) handleFiles(files);
        }

        function handleFileSelect() {
            handleFiles(this.files);
        }

        function handleFiles(files) {
            const file = files[0];
            const isImage = file.type.match('image.*') ||
                file.name.match(/\.(tif|tiff|png|jpg|jpeg)$/i);

            if (!isImage) {
                alert('Please upload an image file (PNG, JPG, or TIFF).');
                return;
            }
            const resultText = resultElement.querySelector(endpoint === 'predict' ? '#result-text' : '#seg-result-text');
            if (resultText) {
                resultText.classList.add('error-message'); // ensure class is applied
                resultText.textContent = ''; // clear message
            }
            //  Reset previous segmentation images (for segment tab)
    if (endpoint === 'segment') {
        if (previewElement) {
            previewElement.src = '';
            previewElement.classList.add('hidden');
        }
        if (maskElement) {
            maskElement.src = '';
            maskElement.classList.add('hidden');
        }
        if (overlayElement) {
            overlayElement.src = '';
            overlayElement.classList.add('hidden');
        }
    }

            const reader = new FileReader();

            if (file.name.match(/\.(tif|tiff)$/i)) {
                reader.onload = function (e) {
                    if (endpoint === 'segment') {
                        previewElement.classList.add('hidden');
                        resultElement.classList.add('hidden');

                        const blob = new Blob([e.target.result], { type: 'image/tiff' });

                        const formData = new FormData();
                        formData.append('image', blob, file.name);
                        segmentImage(formData, resultElement, maskElement, previewElement, overlayElement);
                    } else {
                        const img = new Image();
                        img.onload = function () {
                            const canvas = document.createElement('canvas');
                            canvas.width = img.width;
                            canvas.height = img.height;
                            const ctx = canvas.getContext('2d');
                            ctx.drawImage(img, 0, 0);
                            previewElement.src = canvas.toDataURL('image/png');
                            previewElement.classList.remove('hidden');
                            resultElement.classList.add('hidden');
                            classifyImage(canvas.toDataURL('image/png'), resultElement);
                        };
                        img.src = URL.createObjectURL(new Blob([e.target.result], { type: 'image/png' }));
                    }
                };
                reader.readAsArrayBuffer(file);
            } else {
                reader.onload = function (e) {
                    previewElement.src = e.target.result;
                    previewElement.classList.remove('hidden');
                    resultElement.classList.add('hidden');

                    if (endpoint === 'predict') {
                        classifyImage(e.target.result, resultElement);
                    } else if (endpoint === 'segment') {
                        segmentImage(e.target.result, resultElement, maskElement, previewElement, overlayElement);
                    }
                };
                reader.readAsDataURL(file);
            }
        }
    }

function classifyImage(imageData, resultElement) {
    const resultText = resultElement.querySelector('#result-text');
    const confidenceText = resultElement.querySelector('#confidence');

    resultText.textContent = "Processing...";
    resultText.classList.remove('error-message'); //Remove red styling if any
    confidenceText.textContent = "";
    resultElement.classList.remove('hidden');

    fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: createFormData(imageData)
    })
    .then(async response => {
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            const errorMsg = errorData?.error || `HTTP error! status: ${response.status}`;
            throw new Error(errorMsg);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            resultText.textContent = `Error: ${data.error}`;
            console.error('Classification error:', data.error);
        } else if (data.prediction === 'Not MRI') {
            resultText.textContent = 'Error: This is not a brain MRI.';
        } else {
            resultText.textContent = data.prediction;
            confidenceText.textContent = `Confidence: ${data.confidence.toFixed(2)}`;
            resultText.classList.remove('error-message'); // Remove red styling on success
        }
    })
    .catch(error => {
        resultText.textContent = `Error: ${error.message}`;
        resultText.classList.add('error-message'); //Add red styling on error
        console.error('Classification error:', error);
    });
}

function segmentImage(imageData, resultElement, maskElement, originalElement, overlayElement) {
    const resultText = resultElement.querySelector('#seg-result-text');
    resultText.innerHTML = '<span class="loading"></span> Processing...';
    resultText.classList.remove('error-message'); //Remove red styling
    resultElement.classList.remove('hidden');

    const body = (imageData instanceof FormData) ? imageData : createFormData(imageData);

    fetch('http://localhost:5000/segment', {
        method: 'POST',
        body: body
    })
    .then(async response => {
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            const errorMsg = errorData?.error || `HTTP error! status: ${response.status}`;
            throw new Error(errorMsg);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) throw new Error(data.error);

        originalElement.src = `data:image/png;base64,${data.original_image}`;
        originalElement.classList.remove('hidden');

        maskElement.src = `data:image/png;base64,${data.segmentation_mask}`;
        maskElement.classList.remove('hidden');

        if (overlayElement && data.overlay_image) {
            overlayElement.src = `data:image/png;base64,${data.overlay_image}`;
            overlayElement.classList.remove('hidden');
        }

        resultText.textContent = data.message || "Segmentation complete";
        resultText.classList.remove('error-message'); // Remove red styling on success
    })
    .catch(error => {
        resultText.textContent = `Error: ${error.message}`;
        resultText.classList.add('error-message'); //Add red styling on error
        console.error('Segmentation error:', error);
    });
}

    function createFormData(imageData) {
        const blob = dataURLtoBlob(imageData);
        const formData = new FormData();
        formData.append('image', blob, 'upload.jpg');
        return formData;
    }

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
});
