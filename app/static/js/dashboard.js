document.addEventListener('DOMContentLoaded', async () => {
  try {
    const res = await fetch('/api/dashboard/stats');
    const data = await res.json();

    const labels = data.labels || ['Session 1', 'Session 2', 'Session 3', 'Session 4'];
    const overallScores = data.overall_scores || [68, 74, 81, 87];
    const confidenceTrend = data.confidence_trend || [65, 72, 79, 85];
    const communicationTrend = data.communication_trend || [62, 70, 78, 86];
    const fillerWordsTrend = data.filler_words_trend || [18, 14, 9, 4];

    // Score Progression Line Chart
    const ctxTrend = document.getElementById('scoreTrendChart');
    if (ctxTrend) {
      new Chart(ctxTrend, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Overall Readiness (%)',
              data: overallScores,
              borderColor: '#6366f1',
              backgroundColor: 'rgba(99, 102, 241, 0.15)',
              tension: 0.35,
              fill: true,
              pointBackgroundColor: '#6366f1',
              pointRadius: 4
            },
            {
              label: 'MediaPipe Eye Stability (%)',
              data: confidenceTrend,
              borderColor: '#10b981',
              tension: 0.35,
              fill: false,
              pointBackgroundColor: '#10b981',
              pointRadius: 4
            },
            {
              label: 'Communication Clarity (%)',
              data: communicationTrend,
              borderColor: '#06b6d4',
              tension: 0.35,
              fill: false,
              pointBackgroundColor: '#06b6d4',
              pointRadius: 4
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { 
              labels: { 
                color: '#94a3b8',
                font: { family: 'Google Sans Text', size: 12 }
              } 
            },
            tooltip: {
              backgroundColor: '#181c2a',
              titleColor: '#f8fafc',
              bodyColor: '#94a3b8',
              borderColor: 'rgba(255, 255, 255, 0.1)',
              borderWidth: 1,
              padding: 10
            }
          },
          scales: {
            x: { 
              grid: { color: 'rgba(255, 255, 255, 0.05)' }, 
              ticks: { color: '#94a3b8', font: { family: 'Google Sans Text' } } 
            },
            y: { 
              grid: { color: 'rgba(255, 255, 255, 0.05)' }, 
              ticks: { color: '#94a3b8', font: { family: 'Google Sans Text' } }, 
              min: 40, 
              max: 100 
            }
          }
        }
      });
    }

    // Filler Words Bar Chart
    const ctxFiller = document.getElementById('fillerWordsChart');
    if (ctxFiller) {
      new Chart(ctxFiller, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Filler Words Count',
            data: fillerWordsTrend,
            backgroundColor: 'rgba(245, 158, 11, 0.7)',
            borderColor: '#f59e0b',
            borderWidth: 1,
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { 
              labels: { 
                color: '#94a3b8',
                font: { family: 'Google Sans Text', size: 12 } 
              } 
            }
          },
          scales: {
            x: { 
              grid: { color: 'rgba(255, 255, 255, 0.05)' }, 
              ticks: { color: '#94a3b8', font: { family: 'Google Sans Text' } } 
            },
            y: { 
              grid: { color: 'rgba(255, 255, 255, 0.05)' }, 
              ticks: { color: '#94a3b8', font: { family: 'Google Sans Text' } }, 
              beginAtZero: true 
            }
          }
        }
      });
    }
  } catch (err) {
    console.error('Failed to load dashboard chart stats:', err);
  }
});
