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
# compute Emission Probability
def word_given_tag_emission(word, tag, train_bag = train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)#total number of times the passed tag occurred in train_bag
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    #now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)
    return (count_w_given_tag, count_tag)
# compute  Transition Probability
def t2_given_t1_transition(t2, t1, train_bag = train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)

# Viterbi Heuristic
def Viterbi(words):
    #write your code for viterbi algorithm. return a list of tuple in the form of <word, tag>
    #write your code for viterbi algorithm. return a list of tuple in the form of <word, tag>
    # creating t x t transition matrix of tags, t= no of tags
    # Matrix(i, j) represents P(jth tag after the ith tag)
    tags_matrix = np.zeros((len(T), len(T)), dtype='float32')
    for i, t1 in enumerate(list(T)):
        for j, t2 in enumerate(list(T)): 
            tags_matrix[i, j] = t2_given_t1_transition(t2, t1)[0]/t2_given_t1_transition(t2, t1)[1]
    tags_df = pd.DataFrame(tags_matrix, columns = list(T), index=list(T))
    tags = []
    TagList = list(T)
    
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in TagList:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[tags[-1], tag]
                
            # muliply emission and transition probabilities
            #emission_p = word_given_tag_emission(word, tag)[0]/word_given_tag_emission(word, tag)[1]
            emission_p = word_given_tag_emission(words[key], tag)[0]/word_given_tag_emission(words[key], tag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
            
        pmax = max(p)
        state_max = TagList[p.index(pmax)] 
        tags.append(state_max)
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
    return accuracy*100

def main():
    test_t_words, test_run_base = prepare_sample_test_data()
    print('Viterbi Algorithm Accuracy: ',calculate_accuracy(test_t_words, test_run_base))

if __name__ == '__main__':
    main()
