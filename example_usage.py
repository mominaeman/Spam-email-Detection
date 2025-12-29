"""
Example Usage of the SpamDetector Class

This demonstrates how to use the new SpamDetector class for spam detection.
The class provides a cleaner, object-oriented interface compared to using
individual functions.
"""

import spam_detector as sd

def main():
    print("=" * 70)
    print("SpamDetector Class Usage Example")
    print("=" * 70)
    
    # Step 1: Create a detector instance
    print("\n[Step 1] Creating SpamDetector instance...")
    detector = sd.SpamDetector(delta=0.5)
    print(f"Detector created with delta={detector.delta}")
    
    # Step 2: Train the model
    print("\n[Step 2] Training the model on email dataset...")
    detector.train()
    print(f"Training complete!")
    print(f"  - Vocabulary size: {len(detector.all_uniqueWords)} unique words")
    print(f"  - Prior probabilities: P(spam)={detector.spam_prob:.4f}, P(ham)={detector.ham_prob:.4f}")
    
    # Step 3: Test on various email samples
    print("\n[Step 3] Testing predictions on sample emails...")
    print("-" * 70)
    
    # Test Case 1: Clear spam
    email1 = """
    Subject: WINNER!!! YOU WON $10,000,000!!!
    Congratulations! You are the lucky winner of our grand prize!
    Click here immediately to claim your money! Free cash! Limited time!
    Act now! Don't miss this amazing opportunity! Winner winner!
    """
    
    print("\n📧 Email 1 (Expected: spam):")
    print(f"   Subject: WINNER!!! YOU WON $10,000,000!!!")
    prediction = detector.predict(email1)
    print(f"   Prediction: {prediction.upper()}")
    
    # Test Case 2: Clear ham
    email2 = """
    Subject: Team Meeting Notes - Project Update
    Hi everyone,
    Thanks for attending today's meeting. Here are the key points we discussed:
    1. Project timeline is on track
    2. Next milestone is scheduled for next week
    3. Please review the documentation
    Let me know if you have any questions.
    Best regards,
    John
    """
    
    print("\n📧 Email 2 (Expected: ham):")
    print(f"   Subject: Team Meeting Notes - Project Update")
    prediction = detector.predict(email2)
    print(f"   Prediction: {prediction.upper()}")
    
    # Test Case 3: With detailed scores
    email3 = """
    Subject: Special offer - save money today
    Check out our latest deals and promotions.
    Limited time offer. Visit our website for more information.
    """
    
    print("\n📧 Email 3 (With confidence scores):")
    print(f"   Subject: Special offer - save money today")
    prediction, spam_score, ham_score = detector.predict_with_score(email3)
    print(f"   Prediction: {prediction.upper()}")
    print(f"   Spam score: {spam_score:.2f}")
    print(f"   Ham score: {ham_score:.2f}")
    print(f"   Confidence: {abs(spam_score - ham_score):.2f} (higher is more confident)")
    
    # Step 4: Batch prediction example
    print("\n[Step 4] Batch prediction example...")
    print("-" * 70)
    
    test_emails = [
        ("Free money! Click now! Win prizes!", "spam"),
        ("Meeting at 3pm in conference room B", "ham"),
        ("URGENT! Your account needs verification!", "spam"),
        ("Thanks for your email, I'll get back to you soon", "ham"),
    ]
    
    correct = 0
    for email_text, expected in test_emails:
        prediction = detector.predict(email_text)
        status = "✓" if prediction == expected else "✗"
        correct += (prediction == expected)
        print(f"   {status} '{email_text[:40]}...' → {prediction} (expected: {expected})")
    
    print(f"\n   Accuracy: {correct}/{len(test_emails)} ({100*correct/len(test_emails):.0f}%)")
    
    # Step 5: Demonstrate advantages over functional approach
    print("\n[Step 5] Advantages of the class-based approach:")
    print("-" * 70)
    print("   ✓ Train once, predict many times (no retraining needed)")
    print("   ✓ Clean, intuitive API (detector.predict() vs calling multiple functions)")
    print("   ✓ Encapsulated state (all model parameters in one object)")
    print("   ✓ Error handling (prevents prediction before training)")
    print("   ✓ Multiple instances possible (different models with different parameters)")
    print("   ✓ Easy integration into larger applications")
    
    # Step 6: Multiple detector instances
    print("\n[Step 6] Example: Multiple detectors with different smoothing...")
    print("-" * 70)
    
    detector_low = sd.SpamDetector(delta=0.1)
    detector_high = sd.SpamDetector(delta=1.0)
    
    detector_low.train()
    detector_high.train()
    
    test_email = "Free money! Win big! Click now!"
    
    pred_low, score_low_s, score_low_h = detector_low.predict_with_score(test_email)
    pred_high, score_high_s, score_high_h = detector_high.predict_with_score(test_email)
    
    print(f"   Email: '{test_email}'")
    print(f"   Low smoothing (δ=0.1): {pred_low} (scores: spam={score_low_s:.2f}, ham={score_low_h:.2f})")
    print(f"   High smoothing (δ=1.0): {pred_high} (scores: spam={score_high_s:.2f}, ham={score_high_h:.2f})")
    
    print("\n" + "=" * 70)
    print("Example complete! The SpamDetector class is ready to use.")
    print("=" * 70)

if __name__ == "__main__":
    main()
