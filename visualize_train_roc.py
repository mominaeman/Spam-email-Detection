import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from spam_detector import SpamDetector, text_parser
import glob

# Train the model
# You can adjust delta as needed
detector = SpamDetector(delta=0.5)
detector.train('train/ham', 'train/spam')

def get_score(detector, text):
    words = detector.all_uniqueWords
    email_words = text_parser(text)
    spam_log = 0
    ham_log = 0
    for word in email_words:
        if word in words:
            spam_log += detector.spam_condProb.get(word, 1e-10)
            ham_log += detector.ham_condProb.get(word, 1e-10)
    return spam_log - ham_log

# Prepare true labels and predicted scores
true_labels = []
scores = []

# Ham emails
ham_files = glob.glob('train/ham/*.txt')
for file in ham_files:
    with open(file, 'r', encoding='latin-1', errors='ignore') as f:
        text = f.read()
    true_labels.append(0)  # 0 for ham
    scores.append(get_score(detector, text))

# Spam emails
spam_files = glob.glob('train/spam/*.txt')
for file in spam_files:
    with open(file, 'r', encoding='latin-1', errors='ignore') as f:
        text = f.read()
    true_labels.append(1)  # 1 for spam
    scores.append(get_score(detector, text))

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(true_labels, scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Train Set)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
