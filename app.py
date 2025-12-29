"""
Flask web application for spam email detection with visualizations.
"""

from flask import Flask, render_template, request, jsonify
import spam_detector as sd
import numpy as np
import json
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pickle

app = Flask(__name__)

# Global detector instance
detector = None
evaluation_data = None

def initialize_detector():
    """Initialize and train the spam detector."""
    global detector, evaluation_data
    
    print("Initializing SpamDetector...")
    detector = sd.SpamDetector(delta=0.5)
    detector.train()
    print("Training complete!")
    
    # Generate evaluation metrics
    evaluation_data = generate_evaluation_metrics()
    print("Evaluation metrics generated!")

def generate_evaluation_metrics():
    """Generate comprehensive evaluation metrics for visualization."""
    # Get all test data
    all_trainWords, spam_trainWords, ham_trainWords = sd.trainWord_generator()
    all_uniqueWords = sd.unique_words(all_trainWords)
    spam_bagOfWords, ham_bagOfWords = sd.bagOfWords_genarator(
        all_uniqueWords, spam_trainWords, ham_trainWords
    )
    smoothed_spamBOW, smoothed_hamBOW = sd.smoothed_bagOfWords(
        all_uniqueWords, spam_bagOfWords, ham_bagOfWords, 0.5
    )
    
    nb_of_allEmails = sd.number_of_allEmails()
    nb_of_spamEmails = sd.number_of_spamEmails()
    nb_of_hamEmails = sd.number_of_hamEmails()
    
    spam_prob = sd.spam_probability(nb_of_allEmails, nb_of_spamEmails)
    ham_prob = sd.ham_probability(nb_of_allEmails, nb_of_hamEmails)
    
    spam_condProb = sd.spam_condProbability(
        all_uniqueWords, spam_bagOfWords, smoothed_spamBOW, 0.5
    )
    ham_condProb = sd.ham_condProbability(
        all_uniqueWords, ham_bagOfWords, smoothed_hamBOW, 0.5
    )
    
    # Get scores for all test emails
    ham_score_list, spam_score_list, predicted_label_list, decision_label_list = sd.score_calculator(
        all_uniqueWords, spam_prob, ham_prob, spam_condProb, ham_condProb, 0.5
    )
    
    # Get actual labels
    actual_labels = []
    test_path = sd.test_path
    for directories, subdirectories, files in os.walk(test_path):
        for filename in sorted(files):
            actual_label = "ham" if "ham" in filename else "spam"
            actual_labels.append(actual_label)
    
    # Calculate metrics
    fileNumbers = len(actual_labels)
    
    # Spam metrics
    spam_tp, spam_tn, spam_fp, spam_fn = sd.spamConfusionParams(
        fileNumbers, actual_labels, predicted_label_list
    )
    spam_accuracy = sd.get_spamAccuracy(fileNumbers, actual_labels, predicted_label_list)
    spam_precision = sd.get_spamPrecision(fileNumbers, actual_labels, predicted_label_list)
    spam_recall = sd.get_spamRecall(fileNumbers, actual_labels, predicted_label_list)
    spam_fmeasure = sd.get_spamFmeasure(spam_precision, spam_recall)
    
    # Ham metrics
    ham_tp, ham_tn, ham_fp, ham_fn = sd.hamConfusionParams(
        fileNumbers, actual_labels, predicted_label_list
    )
    ham_accuracy = sd.get_hamAccuracy(fileNumbers, actual_labels, predicted_label_list)
    ham_precision = sd.get_hamPrecision(fileNumbers, actual_labels, predicted_label_list)
    ham_recall = sd.get_hamRecall(fileNumbers, actual_labels, predicted_label_list)
    ham_fmeasure = sd.get_hamFmeasure(ham_precision, ham_recall)
    
    # Prepare data for ROC curve
    y_true = [1 if label == "spam" else 0 for label in actual_labels]
    y_scores = [spam - ham for spam, ham in zip(spam_score_list, ham_score_list)]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    return {
        'spam_metrics': {
            'accuracy': float(spam_accuracy),
            'precision': float(spam_precision),
            'recall': float(spam_recall),
            'f1_score': float(spam_fmeasure),
            'confusion_matrix': {
                'tp': int(spam_tp),
                'tn': int(spam_tn),
                'fp': int(spam_fp),
                'fn': int(spam_fn)
            }
        },
        'ham_metrics': {
            'accuracy': float(ham_accuracy),
            'precision': float(ham_precision),
            'recall': float(ham_recall),
            'f1_score': float(ham_fmeasure),
            'confusion_matrix': {
                'tp': int(ham_tp),
                'tn': int(ham_tn),
                'fp': int(ham_fp),
                'fn': int(ham_fn)
            }
        },
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc)
        },
        'dataset_info': {
            'total_emails': int(nb_of_allEmails),
            'spam_emails': int(nb_of_spamEmails),
            'ham_emails': int(nb_of_hamEmails),
            'vocabulary_size': len(all_uniqueWords),
            'test_emails': len(actual_labels)
        }
    }

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if email is spam or ham."""
    try:
        data = request.get_json()
        email_text = data.get('email', '')
        
        if not email_text.strip():
            return jsonify({'error': 'Email text cannot be empty'}), 400
        
        # Get prediction with scores
        prediction, spam_score, ham_score = detector.predict_with_score(email_text)
        
        # Calculate confidence
        confidence = abs(spam_score - ham_score)
        
        return jsonify({
            'prediction': prediction,
            'spam_score': float(spam_score),
            'ham_score': float(ham_score),
            'confidence': float(confidence),
            'is_spam': prediction == 'spam'
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
        'detector_trained': detector is not None and detector.trained
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
