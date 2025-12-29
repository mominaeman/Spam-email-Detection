"""
Train all three spam detection models in sequence.
Shows progressive improvement from baseline to advanced techniques.
"""

import os
import time
import spam_detector as model1
from spam_detector_enhanced import EnhancedSpamDetector
from spam_detector_tfidf import TFIDFSpamDetector

def train_all_models():
    """Train all three models and show progressive improvement."""
    
    print("\n" + "="*80)
    print(" PROGRESSIVE MODEL TRAINING - ALL THREE MODELS")
    print("="*80)
    print("\nThis will train:")
    print("  1. Model 1: Naive Bayes (Baseline)")
    print("  2. Model 2: Enhanced Features (Model 1 + Statistical Features + SVM)")
    print("  3. Model 3: TF-IDF + Ensemble (Model 2 + TF-IDF + Advanced ML)")
    print("\n" + "="*80 + "\n")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    total_start = time.time()
    
    # ===== Model 1: Naive Bayes =====
    print("─" * 80)
    print("[1/3] TRAINING MODEL 1: NAIVE BAYES (BASELINE)")
    print("─" * 80)
    start_time = time.time()
    
    detector1 = model1.SpamDetector(delta=0.5)
    detector1.train()
    
    # Save Model 1 with proper method
    import pickle
    with open('models/model_naive_bayes.pkl', 'wb') as f:
        pickle.dump({
            'all_uniqueWords': detector1.all_uniqueWords,
            'spam_bagOfWords': detector1.spam_bagOfWords,
            'ham_bagOfWords': detector1.ham_bagOfWords,
            'smoothed_spamBOW': detector1.smoothed_spamBOW,
            'smoothed_hamBOW': detector1.smoothed_hamBOW,
            'spam_prob': detector1.spam_prob,
            'ham_prob': detector1.ham_prob,
            'spam_condProb': detector1.spam_condProb,
            'ham_condProb': detector1.ham_condProb,
            'delta': detector1.delta,
            'trained': detector1.trained
        }, f)
    print("✓ Model 1 saved to: models/model_naive_bayes.pkl")
    
    time1 = time.time() - start_time
    print(f"✓ Model 1 trained in {time1:.2f} seconds")
    print()
    
    # ===== Model 2: Enhanced Features =====
    print("─" * 80)
    print("[2/3] TRAINING MODEL 2: ENHANCED FEATURES")
    print("─" * 80)
    start_time = time.time()
    
    detector2 = EnhancedSpamDetector()
    detector2.train()
    detector2.save_model()
    
    time2 = time.time() - start_time
    print(f"✓ Model 2 trained in {time2:.2f} seconds")
    print()
    
    # ===== Model 3: TF-IDF + Ensemble =====
    print("─" * 80)
    print("[3/3] TRAINING MODEL 3: TF-IDF + ENSEMBLE")
    print("─" * 80)
    start_time = time.time()
    
    detector3 = TFIDFSpamDetector(max_features=1000)
    detector3.train()
    detector3.save_model()
    
    time3 = time.time() - start_time
    print(f"✓ Model 3 trained in {time3:.2f} seconds")
    print()
    
    # ===== Summary =====
    total_time = time.time() - total_start
    
    print("="*80)
    print(" TRAINING COMPLETE - ALL MODELS READY!")
    print("="*80)
    print(f"\n  Model 1 (Naive Bayes):       {time1:>6.2f}s")
    print(f"  Model 2 (Enhanced Features): {time2:>6.2f}s")
    print(f"  Model 3 (TF-IDF + Ensemble): {time3:>6.2f}s")
    print(f"  {'─' * 40}")
    print(f"  Total training time:         {total_time:>6.2f}s")
    print(f"\n  All models saved in: models/")
    print("  • model_naive_bayes.pkl")
    print("  • model_enhanced.pkl")
    print("  • model_tfidf.pkl")
    print("\n" + "="*80)
    print("\n✅ Ready to test! Run 'python compare_models.py' to compare performance")
    print("="*80 + "\n")

if __name__ == "__main__":
    train_all_models()
