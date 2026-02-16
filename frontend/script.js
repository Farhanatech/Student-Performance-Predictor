async function sendData() {
    const resultContainer = document.getElementById('result-container');
    const resultText = document.getElementById('result');
    
    const payload = {
        study_hours: document.getElementById('hours').value,
        attendance: document.getElementById('attendance').value,
        past_scores: document.getElementById('past').value,
        extracurricular: document.getElementById('extra').value
    };

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        
        resultContainer.style.display = 'block';
        resultText.innerText = data.score;
    } catch (error) {
        alert("Backend connect nahi ho raha. Pehle 'python app.py' run karein!");
    }
}