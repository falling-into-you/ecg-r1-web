document.getElementById('inferenceForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submitBtn');
    const loadingDiv = document.getElementById('loading');
    const resultContainer = document.getElementById('resultContainer');
    const resultText = document.getElementById('resultText');
    
    // Reset UI
    submitBtn.disabled = true;
    loadingDiv.classList.remove('hidden');
    resultContainer.classList.add('hidden');
    resultText.textContent = '';
    
    const formData = new FormData(this);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Something went wrong');
        }
        
        resultText.textContent = data.result;
        resultContainer.classList.remove('hidden');
    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Error:', error);
    } finally {
        submitBtn.disabled = false;
        loadingDiv.classList.add('hidden');
    }
});
