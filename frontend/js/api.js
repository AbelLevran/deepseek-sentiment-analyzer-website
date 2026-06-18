const API_BASE_URL = 'http://localhost:8000'; // Change this when deploying

async function predictSingle(text) {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        
        if (!response.ok) throw new Error('API Error');
        return await response.json();
    } catch (error) {
        console.error('Error in predictSingle:', error);
        throw error;
    }
}

async function predictBatch(texts) {
    try {
        const response = await fetch(`${API_BASE_URL}/predict-batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texts })
        });
        
        if (!response.ok) throw new Error('API Error');
        return await response.json();
    } catch (error) {
        console.error('Error in predictBatch:', error);
        throw error;
    }
}

async function getDashboardData() {
    try {
        const response = await fetch(`${API_BASE_URL}/dashboard-data`);
        if (!response.ok) throw new Error('API Error');
        return await response.json();
    } catch (error) {
        console.error('Error in getDashboardData:', error);
        throw error;
    }
}
