import random
import string
from collections import defaultdict


def read_poem(filename):
    try:
        with open(filename, 'r') as file:
            poem = file.read().lower() 
            poem = poem.translate(str.maketrans('', '', string.punctuation))
            poem_lines = poem.split('\n')
            # Split each line into words
            poem_lines = [line.split() for line in poem_lines if line.strip() != '']
            return poem_lines
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []

# initial word probabilities calculation
def calculate_initial_probs(poem_lines):
    initial_probs = defaultdict(int)
    for line in poem_lines:
       
        if len(line) > 0:
            initial_probs[line[0]] += 1
    
    total_lines = len(poem_lines)
 
    for word in initial_probs:
        initial_probs[word] /= total_lines
    return initial_probs


def calculate_first_order_transitions(poem_lines):
    first_order_probs = defaultdict(lambda: defaultdict(int))
    for line in poem_lines:
        for i in range(len(line) - 1):
            first_order_probs[line[i]][line[i + 1]] += 1

    # transiition probabilities normalization 
    for word in first_order_probs:
        total_word_count = sum(first_order_probs[word].values())
        for next_word in first_order_probs[word]:
            first_order_probs[word][next_word] /= total_word_count
    return first_order_probs


def calculate_second_order_transitions(poem_lines):
    second_order_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for line in poem_lines:
        for i in range(len(line) - 2):
            second_order_probs[line[i]][line[i + 1]][line[i + 2]] += 1

    # transiition probabilities normalization 
    for word1 in second_order_probs:
        for word2 in second_order_probs[word1]:
            total_count = sum(second_order_probs[word1][word2].values())
            for word3 in second_order_probs[word1][word2]:
                second_order_probs[word1][word2][word3] /= total_count
    return second_order_probs

# 5. cumulative Probability
def select_word(prob_dist):
    if not prob_dist:  
        return "end"  # Return a default word (e.g., "end") to stop the line generation
    cumulative_probs = []
    total_prob = 0.0
    for word, prob in prob_dist.items():
        total_prob += prob
        cumulative_probs.append((word, total_prob))

    r = random.random()
    for word, cum_prob in cumulative_probs:
        if r < cum_prob:
            return word
    return list(prob_dist.keys())[0]  # Default return if all else fails


def generate_poetry(initial_probs, first_order_probs, second_order_probs, num_lines=4):
    poetry = []
    for _ in range(num_lines):
        line = []
        
        first_word = select_word(initial_probs)
        line.append(first_word)

        
        while len(line) < 10:
            if len(line) == 1:  
                next_word = select_word(first_order_probs[line[-1]])
            elif len(line) > 1:
                prev_word1, prev_word2 = line[-2], line[-1]
                if prev_word1 in second_order_probs and prev_word2 in second_order_probs[prev_word1]:
                    next_word = select_word(second_order_probs[prev_word1][prev_word2])
                else:
                    next_word = select_word(first_order_probs[line[-1]])
            line.append(next_word)


            if next_word == 'end':
                break
        poetry.append(' '.join(line))
    return poetry

def main():
    poem_lines = read_poem('robert_frost.txt')
    if not poem_lines:
        return

    initial_probs = calculate_initial_probs(poem_lines)
    first_order_probs = calculate_first_order_transitions(poem_lines)
    second_order_probs = calculate_second_order_transitions(poem_lines)

    poetry = generate_poetry(initial_probs, first_order_probs, second_order_probs)
    for line in poetry:
        print(line)

if __name__ == '__main__':
    main()
