"""
Test script to verify the SpamDetector class integration.
"""

import spam_detector as sd

# Test 1: Verify SpamDetector class exists
print("=" * 60)
print("Testing SpamDetector Class Integration")
print("=" * 60)

# Test 2: Initialize the detector
print("\n1. Initializing SpamDetector...")
detector = sd.SpamDetector(delta=0.5)
print("   ✓ SpamDetector initialized successfully")
print(f"   - Delta: {detector.delta}")
print(f"   - Trained: {detector.trained}")

# Test 3: Train the model
print("\n2. Training the model...")
try:
    detector.train()
    print("   ✓ Model trained successfully")
    print(f"   - Trained: {detector.trained}")
    print(f"   - Vocabulary size: {len(detector.all_uniqueWords)}")
    print(f"   - Spam probability: {detector.spam_prob:.4f}")
    print(f"   - Ham probability: {detector.ham_prob:.4f}")
except Exception as e:
    print(f"   ✗ Training failed: {e}")

# Test 4: Predict on sample spam email
print("\n3. Testing prediction on spam sample...")
spam_sample = """
Subject: URGENT! You've won $1,000,000!!! 
Click here NOW to claim your prize! Free money! Act fast! Limited time offer!
Congratulations winner! Cash prize awaiting! Click now!
"""

try:
    prediction = detector.predict(spam_sample)
    print(f"   ✓ Prediction: {prediction}")
    
    # Test with scores
    pred, spam_score, ham_score = detector.predict_with_score(spam_sample)
    print(f"   - Spam score: {spam_score:.4f}")
    print(f"   - Ham score: {ham_score:.4f}")
    print(f"   - Predicted as: {pred}")
except Exception as e:
    print(f"   ✗ Prediction failed: {e}")

# Test 5: Predict on sample ham email
print("\n4. Testing prediction on ham sample...")
ham_sample = """
Subject: Project Meeting Tomorrow
Hey team, just a reminder about our project meeting tomorrow at 10 AM.
Please review the documents I sent earlier and come prepared with your updates.
Looking forward to discussing the progress.
"""

try:
    prediction = detector.predict(ham_sample)
    print(f"   ✓ Prediction: {prediction}")
    
    pred, spam_score, ham_score = detector.predict_with_score(ham_sample)
    print(f"   - Spam score: {spam_score:.4f}")
    print(f"   - Ham score: {ham_score:.4f}")
    print(f"   - Predicted as: {pred}")
except Exception as e:
    print(f"   ✗ Prediction failed: {e}")

# Test 6: Verify error handling
print("\n5. Testing error handling...")
try:
    new_detector = sd.SpamDetector()
    new_detector.predict("test")
    print("   ✗ Error handling failed - should have raised exception")
except Exception as e:
    print(f"   ✓ Correctly raised exception: {e}")

# Test 7: Verify existing functions still work
print("\n6. Verifying existing functions still work...")
try:
    nb_all = sd.number_of_allEmails()
    nb_spam = sd.number_of_spamEmails()
    nb_ham = sd.number_of_hamEmails()
    print(f"   ✓ Existing functions work correctly")
    print(f"   - Total emails: {nb_all}")
    print(f"   - Spam emails: {nb_spam}")
    print(f"   - Ham emails: {nb_ham}")
except Exception as e:
    print(f"   ✗ Existing functions broken: {e}")

print("\n" + "=" * 60)
print("SpamDetector Class Integration Tests Complete!")
print("=" * 60)
