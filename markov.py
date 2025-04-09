import pandas as pd
import numpy as np
import re
import math
import random
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# 1. Data Processing
def preprocess_message(message):
    message = message.lower()
    message = re.sub(r'[^a-z\s]', '', message)  
    words = message.split()
    return words

# Load data
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Preprocess
X_train = X_train.apply(preprocess_message)
X_test = X_test.apply(preprocess_message)

# 2. Training
def calculate_initial_and_transition_probs(messages):
    initial_counts = {}
    transition_counts = {}
    total_starts = 0

    for words in messages:
        if not words:
            continue
        first_word = words[0]
        initial_counts[first_word] = initial_counts.get(first_word, 0) + 1
        total_starts += 1
        
        for i in range(len(words) - 1):
            pair = (words[i], words[i+1])
            transition_counts[pair] = transition_counts.get(pair, 0) + 1

    # Laplace smoothing: add-one smoothing
    vocabulary = set()
    for words in messages:
        vocabulary.update(words)
    vocab_size = len(vocabulary)

    # Initial probabilities
    initial_probs = {word: (initial_counts.get(word, 0) + 1) / (total_starts + vocab_size) for word in vocabulary}
    
    # Transition probabilities
    transition_probs = {}
    next_word_counts = {}
    for (w1, w2), count in transition_counts.items():
        next_word_counts[w1] = next_word_counts.get(w1, 0) + count
    for (w1, w2) in transition_counts:
        transition_probs[(w1, w2)] = (transition_counts[(w1, w2)] + 1) / (next_word_counts[w1] + vocab_size)
    
    return initial_probs, transition_probs

# Separate spam and ham messages
spam_messages = [X_train.iloc[i] for i in range(len(y_train)) if y_train.iloc[i] == 'spam']
ham_messages = [X_train.iloc[i] for i in range(len(y_train)) if y_train.iloc[i] == 'ham']

# Calculate priors
num_spam = len(spam_messages)
num_ham = len(ham_messages)
total = num_spam + num_ham
P_spam = num_spam / total
P_ham = num_ham / total
log_P_spam = math.log(P_spam)
log_P_ham = math.log(P_ham)

# Calculate initial and transition probabilities
spam_initial_probs, spam_transition_probs = calculate_initial_and_transition_probs(spam_messages)
ham_initial_probs, ham_transition_probs = calculate_initial_and_transition_probs(ham_messages)

# 3. Prediction
def get_log_likelihood(words, initial_probs, transition_probs):
    if not words:
        return float('-inf')  # Extremely low probability for empty messages

    log_prob = 0.0

    # Initial probability
    prob = initial_probs.get(words[0], 1e-6)
    log_prob += math.log(prob)

    # Transition probabilities
    for i in range(len(words) - 1):
        pair = (words[i], words[i+1])
        prob = transition_probs.get(pair, 1e-6)
        log_prob += math.log(prob)

    return log_prob

def classify_message(words, spam_initial_probs, spam_transition_probs, ham_initial_probs, ham_transition_probs):
    log_P_spam_given_message = get_log_likelihood(words, spam_initial_probs, spam_transition_probs) + log_P_spam
    log_P_ham_given_message = get_log_likelihood(words, ham_initial_probs, ham_transition_probs) + log_P_ham

    if log_P_spam_given_message > log_P_ham_given_message:
        return 'spam'
    else:
        return 'ham'

# Predict on test set
y_pred = [classify_message(message, spam_initial_probs, spam_transition_probs, ham_initial_probs, ham_transition_probs) for message in X_test]

# 4. Evaluation
accuracy = np.mean(np.array(y_pred) == np.array(y_test))
print(f"Accuracy: {accuracy * 100:.2f}%")
