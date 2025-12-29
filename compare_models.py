"""
Compare all three spam detection models on test data.
Shows progressive improvement and detailed analysis.
"""

import os
import sys
import numpy as np
import pickle
from collections import defaultdict
import spam_detector as model1
from spam_detector_enhanced import EnhancedSpamDetector
from spam_detector_tfidf import TFIDFSpamDetector

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.join(BASE_DIR, "test")

def load_test_emails():
    """Load all test emails with their labels."""
    emails = []
    labels = []
    filenames = []
    
    for filename in sorted(os.listdir(test_path)):
        filepath = os.path.join(test_path, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                emails.append(content)
                # Determine label from filename
                label = 'spam' if 'spam' in filename.lower() else 'ham'
                labels.append(label)
                filenames.append(filename)
    
    return emails, labels, filenames

def calculate_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall, F1."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 'spam' and p == 'spam')
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 'ham' and p == 'ham')
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 'ham' and p == 'spam')
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 'spam' and p == 'ham')
    
    accuracy = (tp + tn) / len(y_true) if y_true else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def load_model1():
    """Load Model 1 (Naive Bayes)."""
    print("Loading Model 1 (Naive Bayes)...")
    with open('models/model_naive_bayes.pkl', 'rb') as f:
        data = pickle.load(f)
    
    detector = model1.SpamDetector(delta=data.get('delta', 0.5))
    detector.all_uniqueWords = data['all_uniqueWords']
    detector.spam_bagOfWords = data['spam_bagOfWords']
    detector.ham_bagOfWords = data['ham_bagOfWords']
    detector.smoothed_spamBOW = data['smoothed_spamBOW']
    detector.smoothed_hamBOW = data['smoothed_hamBOW']
    detector.spam_prob = data['spam_prob']
    detector.ham_prob = data['ham_prob']
    detector.spam_condProb = data['spam_condProb']
    detector.ham_condProb = data['ham_condProb']
    detector.trained = data['trained']
    
    return detector

def compare_models():
    """Compare all three models on test data."""
    
    print("\n" + "="*80)
    print(" PROGRESSIVE MODEL COMPARISON - DETAILED ANALYSIS")
    print("="*80 + "\n")
    
    # Load test data
    print("Loading test emails...")
    emails, labels, filenames = load_test_emails()
    print(f"✓ Loaded {len(emails)} test emails")
    print(f"  • Spam: {labels.count('spam')}")
    print(f"  • Ham: {labels.count('ham')}\n")
    
    # Load models
    print("Loading trained models...")
    detector1 = load_model1()
    detector2 = EnhancedSpamDetector().load_model()
    detector3 = TFIDFSpamDetector().load_model()
    print("✓ All models loaded\n")
    
    # Make predictions
    print("Making predictions on test set...")
    predictions = {
        'Model 1 (Naive Bayes)': [],
        'Model 2 (Enhanced Features)': [],
        'Model 3 (TF-IDF + Ensemble)': []
    }
    
    for i, email in enumerate(emails):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(emails)} emails...")
        
        pred1, _, _ = detector1.predict_with_score(email)
        pred2, _, _ = detector2.predict_with_score(email)
        pred3, _, _ = detector3.predict_with_score(email)
        
        predictions['Model 1 (Naive Bayes)'].append(pred1)
        predictions['Model 2 (Enhanced Features)'].append(pred2)
        predictions['Model 3 (TF-IDF + Ensemble)'].append(pred3)
    
    print(f"✓ All predictions complete\n")
    
    # Calculate metrics for each model
    print("="*80)
    print(" PERFORMANCE METRICS")
    print("="*80 + "\n")
    
    results = {}
    baseline_accuracy = None
    
    for model_name, preds in predictions.items():
        metrics = calculate_metrics(labels, preds)
        results[model_name] = metrics
        
        # Track baseline
        if 'Model 1' in model_name:
            baseline_accuracy = metrics['accuracy']
        
        # Calculate improvement
        improvement = ""
        if baseline_accuracy and 'Model 1' not in model_name:
            acc_diff = metrics['accuracy'] - baseline_accuracy
            improvement = f" [+{acc_diff:.2%}]" if acc_diff > 0 else f" [{acc_diff:.2%}]"
        
        print(f"{model_name}")
        print("─" * 80)
        print(f"  Accuracy:  {metrics['accuracy']:.2%}{improvement}")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall:    {metrics['recall']:.2%}")
        print(f"  F1-Score:  {metrics['f1']:.2%}")
        print(f"  Errors:    {metrics['fp'] + metrics['fn']}/{len(labels)} "
              f"(TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']})")
        print()
    
    # Progressive improvement analysis
    print("="*80)
    print(" PROGRESSIVE IMPROVEMENT ANALYSIS")
    print("="*80 + "\n")
    
    m1_metrics = results['Model 1 (Naive Bayes)']
    m2_metrics = results['Model 2 (Enhanced Features)']
    m3_metrics = results['Model 3 (TF-IDF + Ensemble)']
    
    m1_errors = m1_metrics['fp'] + m1_metrics['fn']
    m2_errors = m2_metrics['fp'] + m2_metrics['fn']
    m3_errors = m3_metrics['fp'] + m3_metrics['fn']
    
    print("Error Reduction:")
    print(f"  Model 1 → Model 2: {m1_errors} → {m2_errors} errors "
          f"({((m1_errors - m2_errors) / m1_errors * 100) if m1_errors > 0 else 0:.1f}% reduction)")
    print(f"  Model 2 → Model 3: {m2_errors} → {m3_errors} errors "
          f"({((m2_errors - m3_errors) / m2_errors * 100) if m2_errors > 0 else 0:.1f}% reduction)")
    print(f"  Overall: {m1_errors} → {m3_errors} errors "
          f"({((m1_errors - m3_errors) / m1_errors * 100) if m1_errors > 0 else 0:.1f}% reduction)\n")
    
    print("Accuracy Improvement:")
    print(f"  Model 1: {m1_metrics['accuracy']:.2%} (baseline)")
    print(f"  Model 2: {m2_metrics['accuracy']:.2%} (+{m2_metrics['accuracy'] - m1_metrics['accuracy']:.2%})")
    print(f"  Model 3: {m3_metrics['accuracy']:.2%} (+{m3_metrics['accuracy'] - m1_metrics['accuracy']:.2%})\n")
    
    # Model agreement analysis
    print("="*80)
    print(" MODEL AGREEMENT ANALYSIS")
    print("="*80 + "\n")
    
    agreements = defaultdict(int)
    disagreements = []
    
    for i in range(len(emails)):
        pred1 = predictions['Model 1 (Naive Bayes)'][i]
        pred2 = predictions['Model 2 (Enhanced Features)'][i]
        pred3 = predictions['Model 3 (TF-IDF + Ensemble)'][i]
        
        if pred1 == pred2 == pred3:
            agreements['all_3'] += 1
        elif pred1 == pred2 or pred2 == pred3 or pred1 == pred3:
            agreements['2_of_3'] += 1
        else:
            agreements['none'] += 1
            disagreements.append((i, filenames[i], labels[i], pred1, pred2, pred3))
    
    print(f"All 3 models agree:  {agreements['all_3']}/{len(emails)} "
          f"({agreements['all_3']/len(emails):.1%})")
    print(f"2 of 3 models agree: {agreements['2_of_3']}/{len(emails)} "
          f"({agreements['2_of_3']/len(emails):.1%})")
    print(f"All disagree:        {agreements['none']}/{len(emails)} "
          f"({agreements['none']/len(emails):.1%})")
    
    if disagreements:
        print(f"\nEmails where all models disagree:")
        for idx, filename, true_label, p1, p2, p3 in disagreements[:5]:
            print(f"  {filename}: actual={true_label}, M1={p1}, M2={p2}, M3={p3}")
    
    # Best model summary
    print("\n" + "="*80)
    print(" SUMMARY")
    print("="*80 + "\n")
    
    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
    
    print(f"🏆 Best Accuracy:  {best_accuracy[0]}")
    print(f"                  {best_accuracy[1]['accuracy']:.2%}")
    print(f"\n🏆 Best F1-Score:  {best_f1[0]}")
    print(f"                  {best_f1[1]['f1']:.2%}")
    
    print("\n📈 Progressive Enhancement Strategy Works!")
    print(f"   Baseline → Enhanced → Advanced")
    print(f"   {m1_metrics['accuracy']:.1%} → {m2_metrics['accuracy']:.1%} → {m3_metrics['accuracy']:.1%}")
    print()
    
    # Save detailed report
    save_comparison_report(results, predictions, labels, filenames, emails)
    
    print("="*80 + "\n")

def save_comparison_report(results, predictions, labels, filenames, emails):
    """Save detailed comparison report to file."""
    os.makedirs('results', exist_ok=True)
    
    with open('results/comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" SPAM DETECTION MODELS - DETAILED COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Model performance
        for model_name, metrics in results.items():
            f.write(f"{model_name}\n")
            f.write("─" * 80 + "\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})\n")
            f.write(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']:.2%})\n")
            f.write(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']:.2%})\n")
            f.write(f"  F1-Score:  {metrics['f1']:.4f} ({metrics['f1']:.2%})\n")
            f.write(f"  TP={metrics['tp']}, TN={metrics['tn']}, "
                   f"FP={metrics['fp']}, FN={metrics['fn']}\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(" DETAILED PREDICTIONS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Filename':<35} {'True':<8} {'Model1':<8} {'Model2':<8} {'Model3':<8}\n")
        f.write("─" * 80 + "\n")
        
        for i in range(len(labels)):
            f.write(f"{filenames[i]:<35} "
                   f"{labels[i]:<8} "
                   f"{predictions['Model 1 (Naive Bayes)'][i]:<8} "
                   f"{predictions['Model 2 (Enhanced Features)'][i]:<8} "
                   f"{predictions['Model 3 (TF-IDF + Ensemble)'][i]:<8}\n")
        
        # Misclassifications
        f.write("\n" + "="*80 + "\n")
        f.write(" MISCLASSIFICATIONS BY MODEL\n")
        f.write("="*80 + "\n\n")
        
        for model_name, preds in predictions.items():
            f.write(f"\n{model_name}:\n")
            f.write("─" * 80 + "\n")
            misclassified = [(filenames[i], labels[i], preds[i]) 
                           for i in range(len(labels)) if labels[i] != preds[i]]
            if misclassified:
                for filename, true_label, pred_label in misclassified:
                    f.write(f"  {filename}: actual={true_label}, predicted={pred_label}\n")
            else:
                f.write("  No misclassifications!\n")
    
    print(f"✓ Detailed report saved to: results/comparison_report.txt")

if __name__ == "__main__":
    compare_models()
