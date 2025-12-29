"""
Training script for Model 3: TF-IDF + Ensemble Detector
"""

from spam_detector_tfidf import TFIDFSpamDetector

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" TRAINING MODEL 3: TF-IDF + ENSEMBLE")
    print("="*80 + "\n")
    
    detector = TFIDFSpamDetector(max_features=1000)
    detector.train()
    detector.save_model()
    
    print("\n✓ Training complete!")
    print("✓ Model saved to: models/model_tfidf.pkl")
