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
    setupTabs();
});

function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const performanceSections = document.querySelectorAll('.model-performance');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all buttons and sections
            tabBtns.forEach(b => b.classList.remove('active'));
            performanceSections.forEach(s => s.classList.remove('active'));
            
            // Add active class to clicked button
            btn.classList.add('active');
            
            // Show corresponding section
            const modelId = btn.dataset.model;
            const section = document.getElementById(`${modelId}-performance`);
            if (section) {
                section.classList.add('active');
            }
        });
    });
}

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
    
    // Consensus section
    const consensusBadge = document.getElementById('consensusBadge');
    const consensusText = document.getElementById('consensusText');
    const consensusAgreement = document.getElementById('consensusAgreement');
    
    consensusBadge.className = `consensus-badge ${data.consensus.prediction}`;
    consensusBadge.textContent = data.consensus.prediction.toUpperCase();
    
    const emoji = data.consensus.prediction === 'spam' ? '⚠️' : '✅';
    const message = data.consensus.prediction === 'spam' ? 
        'Models Consensus: This is SPAM' : 
        'Models Consensus: This is LEGITIMATE';
    consensusText.textContent = `${emoji} ${message}`;
    
    const unanimousText = data.consensus.unanimous ? ' (Unanimous)' : '';
    consensusAgreement.textContent = `${data.consensus.agreement} models agree${unanimousText}`;
    
    // Model 1
    displayModelResult('1', data.model1);
    
    // Model 2
    displayModelResult('2', data.model2);
    if (data.model2.detected_features && data.model2.detected_features.length > 0) {
        const features2 = document.getElementById('features2');
        const featuresList = document.getElementById('featuresList');
        featuresList.innerHTML = data.model2.detected_features
            .map(f => `<span class="feature-tag">${f}</span>`)
            .join('');
        features2.style.display = 'block';
    }
    
    // Model 3
    displayModelResult('3', data.model3);
    if (data.model3.top_tfidf_words && data.model3.top_tfidf_words.length > 0) {
        const tfidf3 = document.getElementById('tfidf3');
        const wordsList = document.getElementById('wordsList');
        wordsList.innerHTML = data.model3.top_tfidf_words
            .map(w => `<span class="word-tag">${w}</span>`)
            .join('');
        tfidf3.style.display = 'block';
    }
    
    // Show results
    results.style.display = 'block';
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayModelResult(num, modelData) {
    const resultDiv = document.getElementById(`result${num}`);
    const spamDiv = document.getElementById(`spam${num}`);
    const hamDiv = document.getElementById(`ham${num}`);
    const confDiv = document.getElementById(`conf${num}`);
    
    resultDiv.className = `model-result ${modelData.prediction}`;
    resultDiv.textContent = modelData.prediction.toUpperCase();
    
    spamDiv.textContent = modelData.spam_score.toFixed(3);
    hamDiv.textContent = modelData.ham_score.toFixed(3);
    confDiv.textContent = modelData.confidence.toFixed(3);
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
        
        // Display metrics and charts for each model
        displayModelMetrics('1', data.model1);
        displayModelMetrics('2', data.model2);
        displayModelMetrics('3', data.model3);
        
        // Create comparison charts
        createComparisonCharts(data);
        createDistributionChart(data.dataset_info);
        
    } catch (err) {
        console.error('Failed to load evaluation metrics:', err);
    }
}

function displayModelMetrics(num, modelData) {
    // Display metrics summary
    const metricsContainer = document.getElementById(`metrics${num}`);
    metricsContainer.innerHTML = `
        <div class="metric-item accuracy">
            <span class="label">Accuracy</span>
            <span class="value">${(modelData.accuracy * 100).toFixed(2)}%</span>
        </div>
        <div class="metric-item precision">
            <span class="label">Precision</span>
            <span class="value">${(modelData.precision * 100).toFixed(2)}%</span>
        </div>
        <div class="metric-item recall">
            <span class="label">Recall</span>
            <span class="value">${(modelData.recall * 100).toFixed(2)}%</span>
        </div>
        <div class="metric-item f1">
            <span class="label">F1-Score</span>
            <span class="value">${(modelData.f1_score * 100).toFixed(2)}%</span>
        </div>
    `;
    
    // Create ROC curve
    createROCChart(`rocChart${num}`, modelData.roc_curve, modelData.name);
    
    // Create confusion matrix
    createConfusionMatrix(`confusionChart${num}`, modelData.confusion_matrix, modelData.name);
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

function createROCChart(canvasId, rocData, modelName) {
    const ctx = document.getElementById(canvasId);
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: rocData.fpr.map((_, i) => i),
            datasets: [{
                label: `${modelName} (AUC = ${rocData.auc.toFixed(3)})`,
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

function createConfusionMatrix(canvasId, cm, modelName) {
    const ctx = document.getElementById(canvasId);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['True Positives', 'True Negatives', 'False Positives', 'False Negatives'],
            datasets: [{
                label: 'Count',
                data: [cm.tp, cm.tn, cm.fp, cm.fn],
                backgroundColor: [
                    '#10b981',
                    '#3b82f6',
                    '#f59e0b',
                    '#ef4444'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Emails'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: modelName
                }
            }
        }
    });
}

function createComparisonCharts(data) {
    // ROC Comparison
    const rocCtx = document.getElementById('rocComparison');
    new Chart(rocCtx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: `Model 1 (AUC=${data.model1.roc_curve.auc.toFixed(3)})`,
                    data: data.model1.roc_curve.fpr.map((fpr, i) => ({ 
                        x: fpr, 
                        y: data.model1.roc_curve.tpr[i] 
                    })),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 3,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: `Model 2 (AUC=${data.model2.roc_curve.auc.toFixed(3)})`,
                    data: data.model2.roc_curve.fpr.map((fpr, i) => ({ 
                        x: fpr, 
                        y: data.model2.roc_curve.tpr[i] 
                    })),
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 3,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: `Model 3 (AUC=${data.model3.roc_curve.auc.toFixed(3)})`,
                    data: data.model3.roc_curve.fpr.map((fpr, i) => ({ 
                        x: fpr, 
                        y: data.model3.roc_curve.tpr[i] 
                    })),
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 3,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Random',
                    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                    borderColor: '#94a3b8',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'False Positive Rate' },
                    min: 0,
                    max: 1
                },
                y: {
                    title: { display: true, text: 'True Positive Rate' },
                    min: 0,
                    max: 1
                }
            },
            plugins: {
                legend: { display: true, position: 'bottom' }
            }
        }
    });
    
    // Metrics Comparison
    const metricsCtx = document.getElementById('metricsComparison');
    new Chart(metricsCtx, {
        type: 'bar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            datasets: [
                {
                    label: 'Model 1',
                    data: [
                        data.model1.accuracy * 100,
                        data.model1.precision * 100,
                        data.model1.recall * 100,
                        data.model1.f1_score * 100
                    ],
                    backgroundColor: '#3b82f6'
                },
                {
                    label: 'Model 2',
                    data: [
                        data.model2.accuracy * 100,
                        data.model2.precision * 100,
                        data.model2.recall * 100,
                        data.model2.f1_score * 100
                    ],
                    backgroundColor: '#10b981'
                },
                {
                    label: 'Model 3',
                    data: [
                        data.model3.accuracy * 100,
                        data.model3.precision * 100,
                        data.model3.recall * 100,
                        data.model3.f1_score * 100
                    ],
                    backgroundColor: '#8b5cf6'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: 'Percentage (%)' }
                }
            },
            plugins: {
                legend: { display: true, position: 'bottom' }
            }
        }
    });
    
    // Accuracy Comparison
    const accCtx = document.getElementById('accuracyChart');
    new Chart(accCtx, {
        type: 'bar',
        data: {
            labels: ['Model 1: Naive Bayes', 'Model 2: Enhanced', 'Model 3: TF-IDF'],
            datasets: [{
                label: 'Accuracy (%)',
                data: [
                    data.model1.accuracy * 100,
                    data.model2.accuracy * 100,
                    data.model3.accuracy * 100
                ],
                backgroundColor: ['#3b82f6', '#10b981', '#8b5cf6'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: 'Accuracy (%)' }
                }
            },
            plugins: {
                legend: { display: false }
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
                },
                title: {
                    display: true,
                    text: 'Training Data Distribution'
                }
            }
        }
    });
}
