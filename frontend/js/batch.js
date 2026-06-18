document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBatchBtn');
    const fileInput = document.getElementById('csvFile');
    const textInput = document.getElementById('batchTextInput');
    const loader = document.getElementById('loader');
    const resultsArea = document.getElementById('resultsArea');
    const resultsTable = document.getElementById('resultsTable').querySelector('tbody');
    const downloadBtn = document.getElementById('downloadBtn');

    let currentResults = [];

    analyzeBtn.addEventListener('click', async () => {
        let texts = [];

        if (fileInput.files.length > 0) {
            // Process CSV
            const file = fileInput.files[0];
            Papa.parse(file, {
                header: true,
                skipEmptyLines: true,
                complete: async (results) => {
                    // Try to find the content column
                    let contentCol = '';
                    if (results.meta.fields.includes('content')) {
                        contentCol = 'content';
                    } else if (results.meta.fields.length > 0) {
                        contentCol = results.meta.fields[0]; // fallback to first column
                    }

                    if (!contentCol) {
                        return alert("Could not find a valid text column in CSV.");
                    }

                    texts = results.data.map(row => row[contentCol]).filter(t => t);
                    if(texts.length > 100) {
                        alert("Truncating to 100 texts due to API limits.");
                        texts = texts.slice(0, 100);
                    }
                    await processBatch(texts);
                }
            });
        } else if (textInput.value.trim()) {
            // Process Textarea
            texts = textInput.value.split('\n').map(t => t.trim()).filter(t => t);
            if(texts.length > 100) {
                alert("Truncating to 100 texts due to API limits.");
                texts = texts.slice(0, 100);
            }
            await processBatch(texts);
        } else {
            alert("Please upload a CSV or paste some text.");
        }
    });

    async function processBatch(texts) {
        if(texts.length === 0) return;
        
        analyzeBtn.style.display = 'none';
        loader.style.display = 'block';
        resultsArea.style.display = 'none';
        
        try {
            const result = await predictBatch(texts);
            
            currentResults = texts.map((text, i) => ({
                text: text,
                label: result.results[i].label,
                score: (result.results[i].score * 100).toFixed(1) + '%'
            }));

            renderTable();
            resultsArea.style.display = 'block';
        } catch (error) {
            alert("Failed to process batch. Make sure backend is running.");
        } finally {
            analyzeBtn.style.display = 'inline-block';
            loader.style.display = 'none';
        }
    }

    function renderTable() {
        resultsTable.innerHTML = '';
        currentResults.forEach((item, index) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${index + 1}</td>
                <td style="max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${item.text}</td>
                <td><span class="sentiment-badge ${item.label.toLowerCase()}" style="font-size: 0.8rem; padding: 0.2rem 0.8rem;">${item.label}</span></td>
                <td>${item.score}</td>
            `;
            resultsTable.appendChild(tr);
        });
    }

    downloadBtn.addEventListener('click', () => {
        if(currentResults.length === 0) return;
        
        const csv = Papa.unparse(currentResults);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement("a");
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", "sentiment_results.csv");
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
});
