"""
Training script for Model 2: Enhanced Spam Detector
"""

from spam_detector_enhanced import EnhancedSpamDetector

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" TRAINING MODEL 2: ENHANCED FEATURES")
    print("="*80 + "\n")
    
    detector = EnhancedSpamDetector()
    detector.train()
    detector.save_model()
    
    print("\n✓ Training complete!")
    print("✓ Model saved to: models/model_enhanced.pkl")
