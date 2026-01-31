document.addEventListener('DOMContentLoaded', function() {
    // --- Elements ---
    const form = document.getElementById('inferenceForm');
    const imageInput = document.getElementById('imageInput');
    const ecgInput = document.getElementById('ecgInput');
    const imageDropZone = document.getElementById('imageDropZone');
    const imagePreviewContainer = document.getElementById('imagePreviewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const imageFileName = document.getElementById('imageFileName');
    const removeImageBtn = document.getElementById('removeImageBtn');
    const ecgFileName = document.getElementById('ecgFileName');
    const submitBtn = document.getElementById('submitBtn');
    
    // Result panels
    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const resultContent = document.getElementById('resultContent');
    const diagnosisText = document.getElementById('diagnosisText');
    const reasoningText = document.getElementById('reasoningText');
    const reportDate = document.getElementById('reportDate');

    // --- Helper Function to Parse Result ---

    // --- Image Upload Handling ---
    imageInput.addEventListener('change', function(e) {
        handleImageSelect(this.files[0]);
    });

    // Drag and Drop visual feedback
    imageDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        imageDropZone.classList.add('dragover');
    });

    imageDropZone.addEventListener('dragleave', () => {
        imageDropZone.classList.remove('dragover');
    });

    imageDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        imageDropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            imageInput.files = e.dataTransfer.files; // Update input files
            handleImageSelect(file);
        }
    });

    function handleImageSelect(file) {
        if (file) {
            imageFileName.textContent = file.name;
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreviewContainer.classList.remove('hidden');
            }
            reader.readAsDataURL(file);
        }
    }

    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent triggering dropzone click
        imageInput.value = '';
        imagePreview.src = '';
        imagePreviewContainer.classList.add('hidden');
        imageFileName.textContent = '';
    });

    // --- ECG File Handling ---
    ecgInput.addEventListener('change', function() {
        if (this.files[0]) {
            ecgFileName.textContent = this.files[0].name;
            ecgFileName.style.color = '#0f172a'; // Darker text
        } else {
            ecgFileName.textContent = 'Select .dat/.hea signal file...';
            ecgFileName.style.color = ''; // Reset
        }
    });

    // --- Form Submission ---
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Validate inputs: At least one must be provided
        if (!imageInput.files[0] && !ecgInput.files[0]) {
            alert('Please provide at least one input (Image or ECG Signal File) to proceed.');
            return;
        }

        // UI Updates
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
        
        emptyState.classList.add('hidden');
        resultContent.classList.add('hidden');
        loadingState.classList.remove('hidden');
        
        const formData = new FormData(this);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || 'Analysis failed');
            }
            
            // Success
            loadingState.classList.add('hidden');
            resultContent.classList.remove('hidden');
            
            // Update Report
            reportDate.textContent = new Date().toLocaleDateString('en-US', {
                year: 'numeric', month: 'long', day: 'numeric',
                hour: '2-digit', minute: '2-digit'
            });
            
            // Format text
            const parsed = parseResult(data.result);
            diagnosisText.textContent = parsed.diagnosis;
            reasoningText.textContent = parsed.thinking || "No detailed reasoning process provided by the model.";
            
        } catch (error) {
            loadingState.classList.add('hidden');
            emptyState.classList.remove('hidden'); // Go back to empty state or show error
            alert('Error: ' + error.message);
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<span>Start Analysis</span><i class="fa-solid fa-arrow-right"></i>';
        }
    });
});
