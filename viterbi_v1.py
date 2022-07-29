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
# print(train_tagged_words[:10])
# tokens
tokens = [w_t[0] for w_t in train_tagged_words]
# vocabulary
V = set(tokens)
# number of pos tags in the training corpus
T = set([w_t[1] for w_t in train_tagged_words])
# Create numpy array of no of pos tags by total vocabulary
t = len(T)
v = len(V)



# Viterbi Heuristic
def Viterbi(words):
    #write your code for viterbi algorithm. return a list of tuple in the form of <word, tag>
    all_train_tags = [w_t[1] for w_t in train_tagged_words]
    tags_transition = np.zeros((len(T), len(T))) # DELETE , dtype='float32'
    
    # i = 0
    # j = 0

    for i, tag_i in enumerate(list(T)):
        for j, tag_j in enumerate(list(T)): 
            count_tag_i = len([t for t in T if t==tag_i])
            count_tag_j = 0
            for index in range(len(all_train_tags)-1):
                if all_train_tags[index]==tag_i and all_train_tags[index+1] == tag_j:
                    count_tag_j = count_tag_j + 1
            tags_transition[i,j] = count_tag_j/count_tag_i
            # j = j + 1
        # i = i + 1
    # print(tags_transition)

    df_t = pd.DataFrame(tags_transition, columns = list(T), index=list(T))
    # print(df_t)

    tags = []
    list_T = list(T)
    index = 0
    for word in words:
        # print(word)
        probablity = [] 
        for t in list_T:
            if index == 0:
                probability_of_transition = df_t.loc['.', t]
            else:
                probability_of_transition = df_t.loc[tags[-1], t]
                 
            tag_list = [w_t for w_t in train_tagged_words if w_t[1]==t]
            tag_count = len(tag_list)
            word_tag = [w_t[0] for w_t in tag_list if w_t[0]==word]
            word_tag_count = len(word_tag)
            probablity_of_emission = word_tag_count/tag_count
            probability_of_tag = probablity_of_emission * probability_of_transition    
            probablity.append(probability_of_tag)
             
        max_probability = max(probablity)
        max_tag = list_T[probablity.index(max_probability)] 
        tags.append(max_tag)
        index = index + 1
    return list(zip(words, tags))
    # return

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