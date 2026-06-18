document.addEventListener('DOMContentLoaded', async () => {
    const totalReviewsEl = document.getElementById('totalReviews');
    const statPositif = document.getElementById('statPositif');
    const statNegatif = document.getElementById('statNegatif');
    const statNetral = document.getElementById('statNetral');
    const loader = document.getElementById('loader');
    const dashboardContent = document.getElementById('dashboardContent');
    const ctx = document.getElementById('pieChart').getContext('2d');

    try {
        const data = await getDashboardData();
        
        if (data.error) {
            alert("Error loading dashboard data: " + data.error);
            loader.style.display = 'none';
            return;
        }

        const { total, distribution } = data;
        const p = distribution.Positif || 0;
        const n = distribution.Negatif || 0;
        const net = distribution.Netral || 0;

        totalReviewsEl.textContent = `Total Reviews: ${total.toLocaleString()}`;

        if(total > 0) {
            statPositif.textContent = ((p / total) * 100).toFixed(1) + '%';
            statNegatif.textContent = ((n / total) * 100).toFixed(1) + '%';
            statNetral.textContent = ((net / total) * 100).toFixed(1) + '%';
        }

        // Render Chart
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positif', 'Negatif', 'Netral'],
                datasets: [{
                    data: [p, n, net],
                    backgroundColor: [
                        '#10b981', // success
                        '#ef4444', // danger
                        '#64748b'  // neutral
                    ],
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#f8fafc'
                        }
                    }
                }
            }
        });

        loader.style.display = 'none';
        dashboardContent.style.display = 'block';

    } catch (error) {
        loader.style.display = 'none';
        alert("Could not load dashboard data from backend.");
    }
});
