"""
Flask web application for spam email detection with visualizations.
Supports three progressive models for comparison.
"""

from flask import Flask, render_template, request, jsonify
import spam_detector as sd
from spam_detector_enhanced import EnhancedSpamDetector
from spam_detector_tfidf import TFIDFSpamDetector
import numpy as np
import json
import os
import math
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pickle

app = Flask(__name__)

# Global detector instances for all 3 models
detector1 = None  # Model 1: Naive Bayes
detector2 = None  # Model 2: Enhanced Features
detector3 = None  # Model 3: TF-IDF + Ensemble
evaluation_data = None

def load_model1():
    """Load Model 1 (Naive Bayes)."""
    with open('models/model_naive_bayes.pkl', 'rb') as f:
        data = pickle.load(f)
    
    d = sd.SpamDetector(delta=data.get('delta', 0.5))
    d.all_uniqueWords = data['all_uniqueWords']
    d.spam_bagOfWords = data['spam_bagOfWords']
    d.ham_bagOfWords = data['ham_bagOfWords']
    d.smoothed_spamBOW = data['smoothed_spamBOW']
    d.smoothed_hamBOW = data['smoothed_hamBOW']
    d.spam_prob = data['spam_prob']
    d.ham_prob = data['ham_prob']
    d.spam_condProb = data['spam_condProb']
    d.ham_condProb = data['ham_condProb']
    d.trained = data['trained']
    
    return d

def initialize_detector():
    """Initialize and load all three spam detectors."""
    global detector1, detector2, detector3, evaluation_data
    
    print("Loading all three models...")
    
    # Load Model 1
    print("  [1/3] Loading Model 1 (Naive Bayes)...")
    detector1 = load_model1()
    
    # Load Model 2
    print("  [2/3] Loading Model 2 (Enhanced Features)...")
    detector2 = EnhancedSpamDetector().load_model()
    
    # Load Model 3
    print("  [3/3] Loading Model 3 (TF-IDF + Ensemble)...")
    detector3 = TFIDFSpamDetector().load_model()
    
    print("✓ All models loaded successfully!")
    
    # Generate evaluation metrics for all 3 models
    evaluation_data = generate_all_evaluation_metrics()
    print("✓ Evaluation metrics generated for all models!")

def generate_all_evaluation_metrics():
    """Generate comprehensive evaluation metrics for all 3 models."""
    # Get test data
    test_path = sd.test_path
    test_emails = []
    actual_labels = []
    
    for directories, subdirectories, files in os.walk(test_path):
        for filename in sorted(files):
            filepath = os.path.join(directories, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                test_emails.append(f.read())
            actual_label = "ham" if "ham" in filename else "spam"
            actual_labels.append(actual_label)
    
    # Get dataset info
    nb_of_allEmails = sd.number_of_allEmails()
    nb_of_spamEmails = sd.number_of_spamEmails()
    nb_of_hamEmails = sd.number_of_hamEmails()
    
    all_trainWords, spam_trainWords, ham_trainWords = sd.trainWord_generator()
    all_uniqueWords = sd.unique_words(all_trainWords)
    
    # Generate metrics for each model
    model1_metrics = generate_model_metrics(detector1, test_emails, actual_labels, "Model 1: Naive Bayes")
    model2_metrics = generate_model_metrics(detector2, test_emails, actual_labels, "Model 2: Enhanced")
    model3_metrics = generate_model_metrics(detector3, test_emails, actual_labels, "Model 3: TF-IDF")
    
    return {
        'model1': model1_metrics,
        'model2': model2_metrics,
        'model3': model3_metrics,
        'dataset_info': {
            'total_emails': int(nb_of_allEmails),
            'spam_emails': int(nb_of_spamEmails),
            'ham_emails': int(nb_of_hamEmails),
            'vocabulary_size': len(all_uniqueWords),
            'test_emails': len(actual_labels)
        }
    }

def generate_model_metrics(detector, test_emails, actual_labels, model_name):
    """Generate metrics for a specific model."""
    # Get predictions
    predictions = []
    spam_scores = []
    ham_scores = []
    
    for email in test_emails:
        pred, spam_score, ham_score = detector.predict_with_score(email)
        predictions.append(pred)
        spam_scores.append(spam_score)
        ham_scores.append(ham_score)
    
    # Calculate metrics
    y_true = [1 if label == "spam" else 0 for label in actual_labels]
    y_pred = [1 if pred == "spam" else 0 for pred in predictions]
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Metrics
    total = len(actual_labels)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # ROC curve
    y_scores = [spam - ham for spam, ham in zip(spam_scores, ham_scores)]
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    return {
        'name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        },
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc)
        }
    }

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if email is spam or ham using all three models."""
    try:
        data = request.get_json()
        email_text = data.get('email', '')
        
        if not email_text.strip():
            return jsonify({'error': 'Email text cannot be empty'}), 400
        
        # Get predictions from all 3 models
        pred1, spam1, ham1 = detector1.predict_with_score(email_text)
        pred2, spam2, ham2 = detector2.predict_with_score(email_text)
        pred3, spam3, ham3 = detector3.predict_with_score(email_text)
        
        # Calculate improvements
        accuracy1_baseline = 0.0  # Will be set from evaluation data
        
        # Determine consensus
        predictions = [pred1, pred2, pred3]
        spam_count = predictions.count('spam')
        consensus = 'spam' if spam_count >= 2 else 'ham'
        
        # Get detected features from Model 2
        detected_features = detector2.get_detected_features(email_text)
        
        # Get top TF-IDF words from Model 3
        try:
            top_tfidf_words = detector3.get_top_tfidf_words(email_text, n=10)
        except:
            top_tfidf_words = []
        
        return jsonify({
            'model1': {
                'name': 'Naive Bayes (Baseline)',
                'prediction': pred1,
                'spam_score': float(spam1),
                'ham_score': float(ham1),
                'confidence': float(abs(spam1 - ham1)),
                'is_spam': pred1 == 'spam'
            },
            'model2': {
                'name': 'Enhanced Features + SVM',
                'prediction': pred2,
                'spam_score': float(spam2),
                'ham_score': float(ham2),
                'confidence': float(abs(spam2 - ham2)),
                'is_spam': pred2 == 'spam',
                'detected_features': detected_features
            },
            'model3': {
                'name': 'TF-IDF + Ensemble',
                'prediction': pred3,
                'spam_score': float(spam3),
                'ham_score': float(ham3),
                'confidence': float(abs(spam3 - ham3)),
                'is_spam': pred3 == 'spam',
                'top_tfidf_words': top_tfidf_words
            },
            'consensus': {
                'prediction': consensus,
                'agreement': f"{spam_count}/3" if consensus == 'spam' else f"{3-spam_count}/3",
                'unanimous': spam_count == 3 or spam_count == 0
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluation')
def get_evaluation():
    """Get evaluation metrics."""
    return jsonify(evaluation_data)

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'model1': detector1 is not None and detector1.trained,
            'model2': detector2 is not None and hasattr(detector2, 'model') and detector2.model is not None,
            'model3': detector3 is not None and hasattr(detector3, 'ensemble') and detector3.ensemble is not None
        }
    })

if __name__ == '__main__':
    initialize_detector()
    print("\n" + "="*70)
    print("🚀 Starting Spam Detection Web App")
    print("="*70)
    print("📊 Server running at: http://localhost:5000")
    print("📧 Ready to detect spam emails!")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
