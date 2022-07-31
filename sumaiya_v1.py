import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint, time
 
nltk.download('treebank')
 
nltk.download('universal_tagset')
 
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
 
print(nltk_data[:2])



for sent in nltk_data[:2]:
  for tuple in sent:
    print(tuple)


train_set,test_set =train_test_split(nltk_data,train_size=0.80,test_size=0.20,random_state = 101)



train_tagged_words = [ tup for sent in train_set for tup in sent ]
test_tagged_words = [ tup for sent in test_set for tup in sent ]
print(len(train_tagged_words))
print(len(test_tagged_words))



train_tagged_words[:5]


tags = {tag for word,tag in train_tagged_words}
print(len(tags))
print(tags)
 
vocab = {word for word,tag in train_tagged_words}




def word_given_tag(word, tag, train_bag = train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    count_w_given_tag = len(w_given_tag_list)
 
     
    return (count_w_given_tag, count_tag)


def t2_given_t1(t2, t1, train_bag = train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


 
tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)): 
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]
 
print(tags_matrix)


tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))
display(tags_df)


def Viterbi(words, train_bag = train_tagged_words):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
     
    for key, word in enumerate(words):
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
                 
            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
             
        pmax = max(p)
        state_max = T[p.index(pmax)] 
        state.append(state_max)
    return list(zip(words, state))



random.seed(1234) 
 
rndom = [random.randint(1,len(test_set)) for x in range(10)]
 
test_run = [test_set[i] for i in rndom]
 
test_run_base = [tup for sent in test_run for tup in sent]
 
test_tagged_words = [tup[0] for sent in test_run for tup in sent]


start = time.time()
tagged_seq = Viterbi(test_tagged_words)
end = time.time()
difference = end-start
 
print("Time taken in seconds: ", difference)
 
check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 
 
accuracy = len(check)/len(tagged_seq)
print('Viterbi Algorithm Accuracy: ',accuracy*100)
