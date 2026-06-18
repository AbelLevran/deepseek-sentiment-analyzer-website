document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const reviewInput = document.getElementById('reviewInput');
    const resultWidget = document.getElementById('resultWidget');
    const loader = document.getElementById('loader');
    
    const sentimentLabel = document.getElementById('sentimentLabel');
    const scoreText = document.getElementById('scoreText');
    const scoreBar = document.getElementById('scoreBar');
    const historyTable = document.getElementById('historyTable').querySelector('tbody');

    // Load history from local storage
    const history = JSON.parse(localStorage.getItem('sentimentHistory') || '[]');
    updateHistoryTable();

    analyzeBtn.addEventListener('click', async () => {
        const text = reviewInput.value.trim();
        if (!text) return alert("Please enter some text!");

        analyzeBtn.style.display = 'none';
        loader.style.display = 'block';
        resultWidget.style.display = 'none';

        try {
            const result = await predictSingle(text);
            
            // Update widget
            const scorePercent = (result.score * 100).toFixed(1) + '%';
            sentimentLabel.textContent = result.label;
            scoreText.textContent = scorePercent;
            scoreBar.style.width = scorePercent;

            // Colors
            sentimentLabel.className = 'sentiment-badge ' + result.label.toLowerCase();
            if (result.label.toLowerCase() === 'positif') {
                scoreBar.style.backgroundColor = 'var(--success)';
            } else if (result.label.toLowerCase() === 'negatif') {
                scoreBar.style.backgroundColor = 'var(--danger)';
            } else {
                scoreBar.style.backgroundColor = 'var(--neutral)';
            }

            resultWidget.style.display = 'block';

            // Add to history
            const historyItem = { text: text.substring(0, 50) + (text.length > 50 ? '...' : ''), label: result.label, score: scorePercent };
            history.unshift(historyItem);
            if (history.length > 5) history.pop(); // Keep only last 5
            localStorage.setItem('sentimentHistory', JSON.stringify(history));
            updateHistoryTable();

        } catch (err) {
            alert("Error predicting sentiment. Make sure backend is running.");
        } finally {
            analyzeBtn.style.display = 'inline-block';
            loader.style.display = 'none';
        }
    });

    function updateHistoryTable() {
        historyTable.innerHTML = '';
        history.forEach(item => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${item.text}</td>
                <td><span style="color: var(--${item.label.toLowerCase() === 'positif' ? 'success' : (item.label.toLowerCase() === 'negatif' ? 'danger' : 'neutral')})">${item.label}</span></td>
                <td>${item.score}</td>
            `;
            historyTable.appendChild(tr);
        });
    }
});
