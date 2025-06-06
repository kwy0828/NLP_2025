import re
from collections import Counter, defaultdict

# Sample dataset: simple positive and negative sentences
POSITIVE_SENTENCES = [
    "I love this movie",
    "This was a fantastic experience",
    "What a wonderful day",
]
NEGATIVE_SENTENCES = [
    "I hate this movie",
    "This was a terrible experience",
    "What a horrible day",
]

# Tokenization using regex to split on non-word characters
TOKEN_PATTERN = re.compile(r"\b\w+\b")

def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())

# Build vocabulary and frequency counts for naive Bayes
class NaiveBayesClassifier:
    def __init__(self):
        self.pos_counts = Counter()
        self.neg_counts = Counter()
        self.pos_total = 0
        self.neg_total = 0
        self.vocab = set()

    def train(self, positive_texts, negative_texts):
        for text in positive_texts:
            tokens = tokenize(text)
            self.pos_counts.update(tokens)
            self.pos_total += len(tokens)
            self.vocab.update(tokens)
        for text in negative_texts:
            tokens = tokenize(text)
            self.neg_counts.update(tokens)
            self.neg_total += len(tokens)
            self.vocab.update(tokens)

    def predict(self, text):
        tokens = tokenize(text)
        # Laplace smoothing
        vocab_size = len(self.vocab)
        pos_prob = 0.0
        neg_prob = 0.0
        for token in tokens:
            pos_prob += math.log((self.pos_counts.get(token, 0) + 1) / (self.pos_total + vocab_size))
            neg_prob += math.log((self.neg_counts.get(token, 0) + 1) / (self.neg_total + vocab_size))
        return "positive" if pos_prob >= neg_prob else "negative"

import math

if __name__ == "__main__":
    import sys
    classifier = NaiveBayesClassifier()
    classifier.train(POSITIVE_SENTENCES, NEGATIVE_SENTENCES)
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter a sentence to classify: ")
    label = classifier.predict(text)
    print(f"Sentiment: {label}")
