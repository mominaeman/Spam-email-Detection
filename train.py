import os
import pickle
from collections import Counter

def read_emails(folder):
    emails = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                emails.append(f.read())
    return emails

def tokenize(text):
    return [word.lower() for word in text.split() if word.isalpha()]

def build_vocab(emails):
    vocab = Counter()
    for email in emails:
        vocab.update(tokenize(email))
    return vocab

def main():
    ham_folder = os.path.join('train', 'ham')
    spam_folder = os.path.join('train', 'spam')
    ham_emails = read_emails(ham_folder)
    spam_emails = read_emails(spam_folder)
    print(f"Ham emails: {len(ham_emails)} | Spam emails: {len(spam_emails)}")
    ham_vocab = build_vocab(ham_emails)
    spam_vocab = build_vocab(spam_emails)
    model = {
        'ham_vocab': ham_vocab,
        'spam_vocab': spam_vocab,
        'ham_count': len(ham_emails),
        'spam_count': len(spam_emails)
    }
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Training complete. Model saved as model.pkl.")

if __name__ == "__main__":
    main()
