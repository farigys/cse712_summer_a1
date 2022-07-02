#Importing libraries
import nltk, re, pprint
import numpy as np
import pandas as pd
import pprint, time
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

nltk.download('treebank')
nltk.download('universal_tagset')

# reading the Treebank tagged sentences with universal tagset
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))

random.seed(42)
#Splitting into training and test sets
train_set, test_set = train_test_split(nltk_data, train_size=0.90)
# Getting list of tagged words in training set
train_tagged_words = [tup for sent in train_set for tup in sent]
# tokens
tokens = [pair[0] for pair in train_tagged_words]
# vocabulary
V = set(tokens)
# number of pos tags in the training corpus
T = set([pair[1] for pair in train_tagged_words])
# Create numpy array of no of pos tags by total vocabulary
t = len(T)
v = len(V)


# Viterbi Heuristic
def Viterbi(words):
    #write your code for viterbi algorithm. return a list of tuple in the form of <word, tag>
    return list(zip(words, tags))

def prepare_sample_test_data():
    # list of tagged words in test set
    test_run_base = [tup for sent in test_set for tup in sent]
    # list of  words which are untagged in test set
    test_tagged_words = [tup[0] for sent in test_set for tup in sent]
    test_t_words = test_tagged_words[:100]
    return test_t_words, test_run_base

def calculate_accuracy(test_t_words, test_run_base):
    tagged_seq = Viterbi(test_t_words)
    print(tagged_seq[:5])
    check = [i for i, j in zip(tagged_seq, test_run_base) if i == j]
    accuracy = len(check)/len(tagged_seq)
    return accuracy

def main():
    test_t_words, test_run_base = prepare_sample_test_data()
    print(calculate_accuracy(test_t_words, test_run_base))

if __name__ == '__main__':
    main()