import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from spam_detector import SpamDetector
import glob

# Train the model
# You can adjust delta as needed

detector = SpamDetector(delta=0.5)
detector.train('train/ham', 'train/spam')

# Prepare true and predicted labels for all training emails
true_labels = []
pred_labels = []

# Ham emails
ham_files = glob.glob('train/ham/*.txt')
for file in ham_files:
    with open(file, 'r', encoding='latin-1', errors='ignore') as f:
        text = f.read()
    true_labels.append('ham')
    pred_labels.append(detector.predict(text))

# Spam emails
spam_files = glob.glob('train/spam/*.txt')
for file in spam_files:
    with open(file, 'r', encoding='latin-1', errors='ignore') as f:
        text = f.read()
    true_labels.append('spam')
    pred_labels.append(detector.predict(text))

# Compute confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=['ham', 'spam'])

# Visualize
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Train Set)')
plt.tight_layout()
plt.show()
