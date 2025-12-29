// Sample emails
const samples = {
    spam: `Subject: URGENT! You've Won $1,000,000!!!

Congratulations! You are the LUCKY WINNER of our GRAND PRIZE!

🎉 CLAIM YOUR $1,000,000 NOW! 🎉

Click here IMMEDIATELY to claim your money! This is a LIMITED TIME offer!
FREE CASH! Act now! Don't miss this AMAZING opportunity!

WINNER WINNER! Click the link below:
http://totally-legit-prize.com/claim-now

Hurry! Offer expires soon!`,

    ham: `Subject: Team Meeting Notes - Q4 Planning

Hi Team,

Thanks for attending today's quarterly planning meeting. Here are the key points we discussed:

1. Project Timeline: We're on track to meet our Q4 deadlines
2. Resource Allocation: New team members will join next month
3. Client Feedback: Overall satisfaction scores have improved by 15%
4. Action Items:
   - Review the updated documentation by Friday
   - Submit your Q4 goals by end of week
   - Prepare presentations for the stakeholder meeting

Please let me know if you have any questions or concerns.

Best regards,
Sarah Johnson
Project Manager`
};

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    loadEvaluationMetrics();
    setupEventListeners();
});

function setupEventListeners() {
    const detectBtn = document.getElementById('detectBtn');
    const emailInput = document.getElementById('emailInput');
    const sampleBtns = document.querySelectorAll('.btn-sample');

    detectBtn.addEventListener('click', detectSpam);
    
    emailInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            detectSpam();
        }
    });

    sampleBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const type = e.target.dataset.type;
            emailInput.value = samples[type];
            emailInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
    });
}

async function detectSpam() {
    const emailInput = document.getElementById('emailInput');
    const detectBtn = document.getElementById('detectBtn');
    const results = document.getElementById('results');
    const error = document.getElementById('error');

    const email = emailInput.value.trim();

    if (!email) {
        showError('Please enter email content to analyze');
        return;
    }

    // Show loading state
    detectBtn.disabled = true;
    detectBtn.querySelector('.btn-text').style.display = 'none';
    detectBtn.querySelector('.btn-loader').style.display = 'inline';
    results.style.display = 'none';
    error.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email })
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const data = await response.json();
        displayResults(data);

    } catch (err) {
        showError('An error occurred while analyzing the email. Please try again.');
        console.error(err);
    } finally {
        detectBtn.disabled = false;
        detectBtn.querySelector('.btn-text').style.display = 'inline';
        detectBtn.querySelector('.btn-loader').style.display = 'none';
    }
}

function displayResults(data) {
    const results = document.getElementById('results');
    const resultBadge = document.getElementById('resultBadge');
    const resultText = document.getElementById('resultText');
    const spamScore = document.getElementById('spamScore');
    const hamScore = document.getElementById('hamScore');
    const confidence = document.getElementById('confidence');
    const progressFill = document.getElementById('progressFill');

    // Set badge and text
    resultBadge.className = `badge ${data.prediction}`;
    resultBadge.textContent = data.prediction.toUpperCase();
    
    const emoji = data.is_spam ? '⚠️' : '✅';
    const message = data.is_spam ? 'This looks like SPAM!' : 'This looks like a legitimate email';
    resultText.textContent = `${emoji} ${message}`;

    // Set scores
    spamScore.textContent = data.spam_score.toFixed(2);
    hamScore.textContent = data.ham_score.toFixed(2);
    confidence.textContent = data.confidence.toFixed(2);

    // Calculate progress percentage (0-100, where spam is 0 and ham is 100)
    const total = Math.abs(data.spam_score) + Math.abs(data.ham_score);
    const hamPercentage = (Math.abs(data.ham_score) / total) * 100;
    progressFill.style.width = `${hamPercentage}%`;

    // Show results
    results.style.display = 'block';
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
    const error = document.getElementById('error');
    error.textContent = message;
    error.style.display = 'block';
}

async function loadEvaluationMetrics() {
    try {
        const response = await fetch('/evaluation');
        const data = await response.json();
        
        displayDatasetInfo(data.dataset_info);
        displayMetrics(data);
        createROCChart(data.roc_curve);
        createConfusionMatrix(data.spam_metrics.confusion_matrix);
        createMetricsComparison(data);
        createDistributionChart(data.dataset_info);
        
    } catch (err) {
        console.error('Failed to load evaluation metrics:', err);
    }
}

function displayDatasetInfo(info) {
    const container = document.getElementById('datasetInfo');
    container.innerHTML = `
        <div class="info-item">
            <div class="info-label">Total Training Emails</div>
            <div class="info-value">${info.total_emails}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Spam Emails</div>
            <div class="info-value">${info.spam_emails}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Ham Emails</div>
            <div class="info-value">${info.ham_emails}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Vocabulary Size</div>
            <div class="info-value">${info.vocabulary_size}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Test Emails</div>
            <div class="info-value">${info.test_emails}</div>
        </div>
    `;
}

function displayMetrics(data) {
    displayMetricCard('spamMetrics', data.spam_metrics);
    displayMetricCard('hamMetrics', data.ham_metrics);
}

function displayMetricCard(elementId, metrics) {
    const container = document.getElementById(elementId);
    container.innerHTML = `
        <div class="metric-row">
            <span class="metric-name">Accuracy</span>
            <span class="metric-value">${(metrics.accuracy * 100).toFixed(1)}%</span>
        </div>
        <div class="metric-row">
            <span class="metric-name">Precision</span>
            <span class="metric-value">${(metrics.precision * 100).toFixed(1)}%</span>
        </div>
        <div class="metric-row">
            <span class="metric-name">Recall</span>
            <span class="metric-value">${(metrics.recall * 100).toFixed(1)}%</span>
        </div>
        <div class="metric-row">
            <span class="metric-name">F1 Score</span>
            <span class="metric-value">${(metrics.f1_score * 100).toFixed(1)}%</span>
        </div>
    `;
}

function createROCChart(rocData) {
    const ctx = document.getElementById('rocChart');
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: rocData.fpr.map((_, i) => i),
            datasets: [{
                label: `ROC Curve (AUC = ${rocData.auc.toFixed(3)})`,
                data: rocData.fpr.map((fpr, i) => ({ x: fpr, y: rocData.tpr[i] })),
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.1,
                pointRadius: 0
            }, {
                label: 'Random Classifier',
                data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                borderColor: '#94a3b8',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'False Positive Rate'
                    },
                    min: 0,
                    max: 1
                },
                y: {
                    title: {
                        display: true,
                        text: 'True Positive Rate'
                    },
                    min: 0,
                    max: 1
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom'
                }
            }
        }
    });
}

function createConfusionMatrix(cm) {
    const ctx = document.getElementById('confusionChart');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['True Positive', 'True Negative', 'False Positive', 'False Negative'],
            datasets: [{
                label: 'Count',
                data: [cm.tp, cm.tn, cm.fp, cm.fn],
                backgroundColor: [
                    '#10b981',
                    '#3b82f6',
                    '#f59e0b',
                    '#ef4444'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Emails'
                    }
                }
            }
        }
    });
}

function createMetricsComparison(data) {
    const ctx = document.getElementById('metricsChart');
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            datasets: [{
                label: 'Spam Detection',
                data: [
                    data.spam_metrics.accuracy * 100,
                    data.spam_metrics.precision * 100,
                    data.spam_metrics.recall * 100,
                    data.spam_metrics.f1_score * 100
                ],
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.2)',
                borderWidth: 2
            }, {
                label: 'Ham Detection',
                data: [
                    data.ham_metrics.accuracy * 100,
                    data.ham_metrics.precision * 100,
                    data.ham_metrics.recall * 100,
                    data.ham_metrics.f1_score * 100
                ],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.2)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 20
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function createDistributionChart(info) {
    const ctx = document.getElementById('distributionChart');
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Spam Emails', 'Ham Emails'],
            datasets: [{
                data: [info.spam_emails, info.ham_emails],
                backgroundColor: [
                    '#ef4444',
                    '#10b981'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}
