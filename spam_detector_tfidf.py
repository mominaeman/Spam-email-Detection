"""
Model 3: TF-IDF Spam Detector with Ensemble Learning
Builds on Model 2 by using TF-IDF instead of bag-of-words and ensemble classifiers.
"""

import os
import re
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from spam_detector_enhanced import EnhancedSpamDetector

# Setting paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, "train")
ham_train_path = os.path.join(train_path, "ham")
spam_train_path = os.path.join(train_path, "spam")
test_path = os.path.join(BASE_DIR, "test")

# Stop words (from Model 1)
STOP_WORDS = {
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with',
    'to', 'for', 'of', 'as', 'by', 'that', 'this', 'it', 'from', 'be', 'are', 'was',
    'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'can', 'am', 'i', 'you', 'he', 'she', 'we',
    'they', 'what', 'when', 'where', 'who', 'why', 'how', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'if', 'into'
}


class TFIDFSpamDetector:
    """
    Advanced spam detector using TF-IDF, N-grams, statistical features, and ensemble learning.
    Builds on Model 2 by upgrading word features to TF-IDF and using ensemble classifiers.
    """
    
    def __init__(self, max_features=1000):
        """
        Initialize TF-IDF detector.
        
        Args:
            max_features: Maximum number of TF-IDF features to use
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=list(STOP_WORDS),
            min_df=2,  # Word must appear in at least 2 documents
            max_df=0.8,  # Word must not appear in more than 80% of documents
            ngram_range=(1, 2),  # Unigrams and bigrams
            sublinear_tf=True,  # Use logarithmic TF scaling
            norm='l2'
        )
        self.model = None
        self.trained = False
        
        # Model 2 component for statistical features
        self.stat_extractor = EnhancedSpamDetector()
    
    def preprocess_text(self, text):
        """
        Preprocess text for TF-IDF.
        
        Args:
            text: Email content
            
        Returns:
            str: Preprocessed text
        """
        text = text.lower()
        # Keep only alphabetic words with 3+ letters
        words = re.findall(r'\b[a-z]{3,}\b', text)
        return ' '.join(words)
    
    def load_emails_from_directory(self, directory, label):
        """
        Load and preprocess emails from directory.
        
        Args:
            directory: Path to email directory
            label: 0 for ham, 1 for spam
            
        Returns:
            tuple: (texts, labels, raw_contents)
        """
        texts = []
        labels = []
        raw_contents = []
        
        for filename in sorted(os.listdir(directory)):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    processed = self.preprocess_text(content)
                    texts.append(processed)
                    labels.append(label)
                    raw_contents.append(content)
        
        return texts, labels, raw_contents
    
    def train(self, ham_path=None, spam_path=None):
        """
        Train the TF-IDF + Ensemble model.
        
        Args:
            ham_path: Path to ham emails (optional)
            spam_path: Path to spam emails (optional)
        """
        ham_path = ham_path or ham_train_path
        spam_path = spam_path or spam_train_path
        
        print("Loading and preprocessing emails...")
        ham_texts, ham_labels, ham_raw = self.load_emails_from_directory(ham_path, 0)
        spam_texts, spam_labels, spam_raw = self.load_emails_from_directory(spam_path, 1)
        
        # Combine data
        X_text = ham_texts + spam_texts
        X_raw = ham_raw + spam_raw
        y = np.array(ham_labels + spam_labels)
        
        print(f"Total training samples: {len(X_text)}")
        print(f"  Ham: {len(ham_texts)}")
        print(f"  Spam: {len(spam_texts)}")
        
        # Compute TF-IDF features
        print("\nComputing TF-IDF features...")
        X_tfidf = self.vectorizer.fit_transform(X_text)
        print(f"✓ TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"✓ TF-IDF matrix shape: {X_tfidf.shape}")
        
        # Get top TF-IDF features
        self._print_top_tfidf_features(10)
        
        # Extract statistical features using Model 2's approach
        print("\nExtracting statistical features from Model 2...")
        stat_features = []
        for raw_content in X_raw:
            stat_feat = self.stat_extractor.extract_statistical_features(raw_content)
            stat_features.append(stat_feat)
        stat_features = np.array(stat_features)
        print(f"✓ Statistical features shape: {stat_features.shape}")
        
        # Combine TF-IDF and statistical features
        print("\nCombining TF-IDF and statistical features...")
        X_combined = np.hstack([X_tfidf.toarray(), stat_features])
        print(f"✓ Combined feature shape: {X_combined.shape}")
        
        # Create ensemble classifier
        print("\nTraining ensemble classifier (Random Forest + Gradient Boosting + SVM)...")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        svm = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('svm', svm)
            ],
            voting='soft',  # Use probability averaging
            n_jobs=-1
        )
        
        print("Training (this may take 1-2 minutes)...")
        self.model.fit(X_combined, y)
        
        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X_combined, y, cv=5, scoring='f1', n_jobs=-1)
        print(f"✓ Cross-validation F1 scores: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"✓ Mean CV F1 score: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        
        self.trained = True
    
    def _print_top_tfidf_features(self, n=10):
        """Print top N TF-IDF features."""
        feature_names = self.vectorizer.get_feature_names_out()
        print(f"\nTop {n} TF-IDF features (words/bigrams):")
        for i, name in enumerate(feature_names[:n]):
            print(f"  {i+1}. '{name}'")
    
    def predict(self, email_text):
        """
        Predict if email is spam or ham.
        
        Args:
            email_text: Email content
            
        Returns:
            str: 'spam' or 'ham'
        """
        if not self.trained:
            raise Exception("Model not trained. Call train() first.")
        
        # Preprocess and get TF-IDF features
        processed = self.preprocess_text(email_text)
        tfidf_features = self.vectorizer.transform([processed]).toarray()
        
        # Get statistical features
        stat_features = self.stat_extractor.extract_statistical_features(email_text).reshape(1, -1)
        
        # Combine
        combined = np.hstack([tfidf_features, stat_features])
        
        # Predict
        prediction = self.model.predict(combined)[0]
        
        return 'spam' if prediction == 1 else 'ham'
    
    def predict_with_score(self, email_text):
        """
        Predict with probability scores.
        
        Args:
            email_text: Email content
            
        Returns:
            tuple: (prediction, spam_probability, ham_probability)
        """
        if not self.trained:
            raise Exception("Model not trained. Call train() first.")
        
        # Preprocess and get TF-IDF features
        processed = self.preprocess_text(email_text)
        tfidf_features = self.vectorizer.transform([processed]).toarray()
        
        # Get statistical features
        stat_features = self.stat_extractor.extract_statistical_features(email_text).reshape(1, -1)
        
        # Combine
        combined = np.hstack([tfidf_features, stat_features])
        
        # Predict probabilities
        probabilities = self.model.predict_proba(combined)[0]
        ham_prob = probabilities[0]
        spam_prob = probabilities[1]
        
        prediction = 'spam' if spam_prob > ham_prob else 'ham'
        
        return prediction, spam_prob, ham_prob
    
    def get_top_tfidf_words(self, email_text, n=10):
        """
        Get top N words contributing to TF-IDF score.
        
        Args:
            email_text: Email content
            n: Number of top words to return
            
        Returns:
            list: [(word, tfidf_score), ...]
        """
        if not self.trained:
            raise Exception("Model not trained. Call train() first.")
        
        processed = self.preprocess_text(email_text)
        tfidf_vector = self.vectorizer.transform([processed])
        
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_vector.toarray()[0]
        
        # Get non-zero features
        word_scores = [(feature_names[i], tfidf_scores[i]) 
                      for i in range(len(tfidf_scores)) if tfidf_scores[i] > 0]
        
        # Sort by score
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        return word_scores[:n]
    
    def save_model(self, filepath='models/model_tfidf.pkl'):
        """Save trained model to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'stat_extractor': self.stat_extractor,
                'trained': self.trained,
                'max_features': self.max_features
            }, f)
        print(f"✓ Model 3 saved to {filepath}")
    
    def load_model(self, filepath='models/model_tfidf.pkl'):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.stat_extractor = data['stat_extractor']
            self.trained = data['trained']
            self.max_features = data.get('max_features', 1000)
        print(f"✓ Model 3 loaded from {filepath}")
        return self


if __name__ == "__main__":
    print("="*80)
    print(" MODEL 3: TF-IDF + ENSEMBLE SPAM DETECTOR")
    print("="*80)
    print("\nThis model builds on Model 2 by adding:")
    print("  • TF-IDF weighting instead of raw word counts")
    print("  • Bigrams (2-word phrases) in addition to single words")
    print("  • Statistical features from Model 2")
    print("  • Ensemble voting (Random Forest + Gradient Boosting + SVM)")
    print("\n" + "="*80 + "\n")
    
    detector = TFIDFSpamDetector(max_features=1000)
    detector.train()
    detector.save_model()
    
    # Test on a sample
    print("\n" + "="*80)
    print(" TESTING MODEL 3")
    print("="*80 + "\n")
    
    test_spam = """Subject: URGENT! You've Won $1,000,000!!!

CLICK HERE NOW to claim your FREE CASH PRIZE!
Act now! Limited time offer! Don't miss out!
http://totally-legit-prize.com

WINNER WINNER! FREE MONEY! HURRY!"""
    
    print("Test Email (Spam):")
    print("-" * 80)
    print(test_spam)
    print("-" * 80)
    
    prediction, spam_prob, ham_prob = detector.predict_with_score(test_spam)
    top_words = detector.get_top_tfidf_words(test_spam, n=5)
    
    print(f"\nPrediction: {prediction.upper()}")
    print(f"Spam Probability: {spam_prob:.2%}")
    print(f"Ham Probability: {ham_prob:.2%}")
    
    print("\nTop TF-IDF contributing words:")
    for word, score in top_words:
        print(f"  • '{word}': {score:.4f}")
    
    print("\n" + "="*80)
    print("✓ Model 3 ready!")
    print("="*80)
