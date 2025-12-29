"""
Model 2: Enhanced Spam Detector with Statistical Features
Builds on Model 1 by adding hand-crafted features on top of bag-of-words.
"""

import os
import re
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import spam_detector as model1

# Setting paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, "train")
ham_train_path = os.path.join(train_path, "ham")
spam_train_path = os.path.join(train_path, "spam")
test_path = os.path.join(BASE_DIR, "test")

# Spam keywords
SPAM_KEYWORDS = [
    'free', 'win', 'winner', 'cash', 'prize', 'bonus', 'urgent', 'act now',
    'limited time', 'click here', 'buy now', 'order now', 'subscribe',
    'million', 'dollars', 'money', 'credit', 'loan', 'debt', 'investment',
    'guarantee', 'no risk', 'discount', 'cheap', 'lowest price', 'offer',
    'call now', 'don\'t delete', 'once in lifetime', 'dear friend',
    'congratulations', 'selected', 'claim', 'verify', 'account', 'password',
    'winner', 'won', 'winning', 'collect', 'beneficiary', 'inheritance'
]

URGENT_WORDS = ['urgent', 'immediate', 'immediately', 'now', 'today', 'hurry', 
                'fast', 'quick', 'asap', 'limited', 'expire', 'expires']


class EnhancedSpamDetector:
    """
    Enhanced spam detector combining bag-of-words with statistical features.
    Uses SVM classifier on combined feature set.
    """
    
    def __init__(self):
        """Initialize the enhanced detector."""
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False
        
        # Model 1 components (bag-of-words)
        self.all_uniqueWords = None
        self.spam_bagOfWords = None
        self.ham_bagOfWords = None
        
    def extract_statistical_features(self, text):
        """
        Extract 18 hand-crafted statistical features from email text.
        
        Args:
            text: Email content as string
            
        Returns:
            np.array: 18-dimensional feature vector
        """
        text_lower = text.lower()
        features = []
        
        # Split into words
        words = text.split()
        
        # 1. Basic text statistics
        features.append(len(text))  # Character count
        features.append(len(words))  # Word count
        features.append(np.mean([len(w) for w in words]) if words else 0)  # Avg word length
        features.append(text.count('\n'))  # Line count
        
        # 2. Capitalization features
        features.append(sum(1 for c in text if c.isupper()) / len(text) if text else 0)  # Capital ratio
        features.append(sum(1 for w in words if w.isupper() and len(w) > 1))  # ALL CAPS words
        features.append(sum(1 for w in words if w and w[0].isupper()) / len(words) if words else 0)  # Capitalized ratio
        
        # 3. Punctuation features
        features.append(text.count('!'))  # Exclamation marks
        features.append(text.count('?'))  # Question marks
        features.append((text.count('!') + text.count('?')) / len(words) if words else 0)  # Punctuation per word
        
        # 4. Money and number features
        features.append(text.count('$') + text.count('€') + text.count('£') + text.count('¥'))  # Currency symbols
        features.append(len(re.findall(r'\d+', text)))  # Number count
        features.append(sum(c.isdigit() for c in text) / len(text) if text else 0)  # Digit ratio
        
        # 5. URL and email features
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        features.append(len(urls))  # URL count
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        features.append(len(emails))  # Email address count
        
        # 6. Spam keywords
        spam_keyword_count = sum(1 for keyword in SPAM_KEYWORDS if keyword in text_lower)
        features.append(spam_keyword_count)  # Spam keyword count
        
        # 7. Urgent words
        urgent_count = sum(1 for word in URGENT_WORDS if word in text_lower)
        features.append(urgent_count)  # Urgent word count
        
        # 8. Special patterns
        features.append(1 if re.search(r'click\s+here', text_lower) else 0)  # "Click here" pattern
        
        return np.array(features)
    
    def get_bow_features(self, text):
        """
        Get bag-of-words features using Model 1's vocabulary.
        
        Args:
            text: Email content
            
        Returns:
            np.array: BOW feature vector
        """
        if self.all_uniqueWords is None:
            raise Exception("Model not trained. No vocabulary available.")
        
        # Parse text using Model 1's parser
        words = model1.text_parser(text)
        
        # Create bag-of-words vector
        bow_vector = np.zeros(len(self.all_uniqueWords))
        for i, word in enumerate(self.all_uniqueWords):
            bow_vector[i] = words.count(word)
        
        return bow_vector
    
    def extract_combined_features(self, text):
        """
        Extract combined features: BOW + statistical features.
        
        Args:
            text: Email content
            
        Returns:
            np.array: Combined feature vector
        """
        # Get BOW features
        bow_features = self.get_bow_features(text)
        
        # Get statistical features
        stat_features = self.extract_statistical_features(text)
        
        # Concatenate
        combined = np.concatenate([bow_features, stat_features])
        
        return combined
    
    def train(self, ham_path=None, spam_path=None):
        """
        Train the enhanced model with combined features.
        
        Args:
            ham_path: Path to ham emails (optional)
            spam_path: Path to spam emails (optional)
        """
        ham_path = ham_path or ham_train_path
        spam_path = spam_path or spam_train_path
        
        print("Building vocabulary using Model 1's approach...")
        # Use Model 1's functions to build vocabulary
        all_trainWords, spam_trainWords, ham_trainWords = model1.trainWord_generator()
        self.all_uniqueWords = model1.unique_words(all_trainWords)
        self.spam_bagOfWords, self.ham_bagOfWords = model1.bagOfWords_genarator(
            self.all_uniqueWords, spam_trainWords, ham_trainWords
        )
        
        print(f"Vocabulary size: {len(self.all_uniqueWords)} words")
        
        # Load all emails and extract combined features
        print("Extracting combined features from ham emails...")
        ham_features = []
        for filename in sorted(os.listdir(ham_path)):
            filepath = os.path.join(ham_path, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    features = self.extract_combined_features(content)
                    ham_features.append(features)
        
        print("Extracting combined features from spam emails...")
        spam_features = []
        for filename in sorted(os.listdir(spam_path)):
            filepath = os.path.join(spam_path, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    features = self.extract_combined_features(content)
                    spam_features.append(features)
        
        # Combine data
        X = np.array(ham_features + spam_features)
        y = np.array([0] * len(ham_features) + [1] * len(spam_features))
        
        print(f"Total training samples: {len(X)}")
        print(f"Feature dimensions: {X.shape[1]} (BOW: {len(self.all_uniqueWords)}, Statistical: 18)")
        
        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Train SVM with grid search
        print("Training SVM with grid search (this may take a minute)...")
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.01],
            'kernel': ['rbf']
        }
        
        svm = SVC(probability=True, random_state=42)
        self.model = GridSearchCV(svm, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        self.model.fit(X_scaled, y)
        
        print(f"\n✓ Best parameters: {self.model.best_params_}")
        print(f"✓ Best cross-validation F1 score: {self.model.best_score_:.4f}")
        
        self.trained = True
    
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
        
        features = self.extract_combined_features(email_text).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        
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
        
        features = self.extract_combined_features(email_text).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        probabilities = self.model.predict_proba(features_scaled)[0]
        ham_prob = probabilities[0]
        spam_prob = probabilities[1]
        
        prediction = 'spam' if spam_prob > ham_prob else 'ham'
        
        return prediction, spam_prob, ham_prob
    
    def get_detected_features(self, email_text):
        """
        Get human-readable feature analysis.
        
        Args:
            email_text: Email content
            
        Returns:
            dict: Detected features
        """
        stat_features = self.extract_statistical_features(email_text)
        text_lower = email_text.lower()
        
        detected_keywords = [kw for kw in SPAM_KEYWORDS if kw in text_lower]
        detected_urgent = [uw for uw in URGENT_WORDS if uw in text_lower]
        
        urls = re.findall(r'http[s]?://[^\s]+', email_text)
        
        return {
            'capital_ratio': float(stat_features[4]),
            'all_caps_words': int(stat_features[5]),
            'exclamation_marks': int(stat_features[7]),
            'question_marks': int(stat_features[8]),
            'currency_symbols': int(stat_features[10]),
            'urls': urls,
            'url_count': int(stat_features[13]),
            'spam_keywords': detected_keywords,
            'urgent_words': detected_urgent,
            'has_click_here': bool(stat_features[17])
        }
    
    def save_model(self, filepath='models/model_enhanced.pkl'):
        """Save trained model to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'trained': self.trained,
                'all_uniqueWords': self.all_uniqueWords,
                'spam_bagOfWords': self.spam_bagOfWords,
                'ham_bagOfWords': self.ham_bagOfWords
            }, f)
        print(f"✓ Model 2 saved to {filepath}")
    
    def load_model(self, filepath='models/model_enhanced.pkl'):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.trained = data['trained']
            self.all_uniqueWords = data['all_uniqueWords']
            self.spam_bagOfWords = data.get('spam_bagOfWords')
            self.ham_bagOfWords = data.get('ham_bagOfWords')
        print(f"✓ Model 2 loaded from {filepath}")
        return self


if __name__ == "__main__":
    print("="*80)
    print(" MODEL 2: ENHANCED SPAM DETECTOR (BOW + Statistical Features + SVM)")
    print("="*80)
    print("\nThis model builds on Model 1 by adding:")
    print("  • 18 hand-crafted statistical features")
    print("  • Capital letter analysis")
    print("  • URL and email detection")
    print("  • Spam keyword matching")
    print("  • SVM classifier instead of Naive Bayes")
    print("\n" + "="*80 + "\n")
    
    detector = EnhancedSpamDetector()
    detector.train()
    detector.save_model()
    
    # Test on a sample
    print("\n" + "="*80)
    print(" TESTING MODEL 2")
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
    features = detector.get_detected_features(test_spam)
    
    print(f"\nPrediction: {prediction.upper()}")
    print(f"Spam Probability: {spam_prob:.2%}")
    print(f"Ham Probability: {ham_prob:.2%}")
    
    print("\nDetected Features:")
    print(f"  • Capital ratio: {features['capital_ratio']:.1%}")
    print(f"  • ALL CAPS words: {features['all_caps_words']}")
    print(f"  • Exclamation marks: {features['exclamation_marks']}")
    print(f"  • Currency symbols: {features['currency_symbols']}")
    print(f"  • URLs: {features['url_count']}")
    print(f"  • Spam keywords: {', '.join(features['spam_keywords'][:5])}")
    print(f"  • Urgent words: {', '.join(features['urgent_words'][:3])}")
    print(f"  • Has 'click here': {features['has_click_here']}")
    
    print("\n" + "="*80)
    print("✓ Model 2 ready!")
    print("="*80)
