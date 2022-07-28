#Importing libraries
import nltk, re, pprint
import numpy as np
import pandas as pd
import pprint, time
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from tqdm import tqdm,tqdm_pandas

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

# transition probability
# probability of tag2 given tag1 
def transition_probability_calculator(t2,t1,train_tags = train_tagged_words):
    tags = [tup[1] for tup in train_tagged_words]
    t1_count = len([t for t in tags if t==t1])
    count_t2_given_t1= 0
    
    # we are checking the next index in each iteration.
    # that's why range is decreased by 1
    for i in range(len(tags)-1):
        if tags[i]==t1 and tags[i+1]==t2:
            count_t2_given_t1+=1
    transition_probability = count_t2_given_t1/t1_count
    return transition_probability

# Emission probability
# word given tag
def emission_probability_calculator(word,tag, train_tags = train_tagged_words):
    given_tag_list = [tup for tup in train_tags if tup[1]==tag]
    count_given_tag_list = len(given_tag_list)
    
    word_given_tag_list = [tup[0] for tup in given_tag_list if tup[0] == word]
    count_word_given_tag_list = len(word_given_tag_list)
    
    emission_probability = count_word_given_tag_list/count_given_tag_list
    
    return emission_probability

# Viterbi Heuristic
def Viterbi(words):
    #write your code for viterbi algorithm. return a list of tuple in the form of <word, tag>
    
    #calculating transition probability matrix : probablity of tag2 given tag1
    transition_prob_matrix = np.zeros((len(T),len(T)),dtype='float32')
    for i,t1 in enumerate(list(T)):
        for j,t2 in enumerate(list(T)):
            transition_prob_matrix[i,j]=transition_probability_calculator(t2,t1)


    transition_probability_matrix_df = pd.DataFrame(transition_prob_matrix,columns=list(T),index=list(T))
    
    #-------------------------------------------------------------------------
    tags = []
    tagset_list=list(T)
    for key,word in tqdm(enumerate(words)):
        
        # state probability list for each given observaion
        tag_probability_list_for_given_word =[]
        for tag in tagset_list:
            if key==0:
                transition_probability = transition_probability_matrix_df.loc['.',tag]
            else:
                transition_probability = transition_probability_matrix_df.loc[tags[-1],tag]
            
            # calculating emission probability
            emission_probability = emission_probability_calculator(word,tag)
            # calculating state probability of a word given tag
            tag_state_probability = emission_probability * transition_probability
            # adding tag_state_probability for a word given tag to the tag_probability_list, for all the tags
            tag_probability_list_for_given_word.append(tag_state_probability)
            
        #selecting the tag with highest state probability for a word
        max_tag_probability_for_given_word= max(tag_probability_list_for_given_word)
        tag_max_probability = tagset_list[tag_probability_list_for_given_word.index(max_tag_probability_for_given_word)]
        tags.append(tag_max_probability)    
        
    
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