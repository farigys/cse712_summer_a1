@@ -1,57 +1,341 @@
#Importing libraries
import nltk, re, pprint
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#CSE712 (sec-01)
#Sadia Afrose - 21266004
#(Please Find Both number 1 and 2 below; they are both togather here)


# In[ ]:


#Number 2


# In[106]:


import glob
import re
from collections import Counter
from nltk.corpus.reader import TaggedCorpusReader
from collections import defaultdict


# In[107]:


# Importing libraries
import nltk
import numpy as np
import pandas as pd
import pprint, time
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

import pprint, time

#download the treebank corpus from nltk
nltk.download('treebank')

#download the universal tagset from nltk
nltk.download('universal_tagset')

# reading the Treebank tagged sentences with universal tagset
 
# reading the Treebank tagged sentences
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))

#print the first two sentences along with tags
print(nltk_data[:2])


# In[108]:


for sent in nltk_data[:2]:
  for tuple in sent:
    print(tuple)


# In[109]:


train_set,test_set =train_test_split(nltk_data,train_size=0.80,test_size=0.20,random_state = 101)


# In[110]:


train_tagged_words = [ tup for sent in train_set for tup in sent ]
test_tagged_words = [ tup for sent in test_set for tup in sent ]
print(len(train_tagged_words))
print(len(test_tagged_words))


# In[111]:


train_tagged_words[:5]

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

# In[112]:


tags = {tag for word,tag in train_tagged_words}
print(len(tags))
print(tags)

# check total words in vocabulary
vocab = {word for word,tag in train_tagged_words}


# In[113]:


def word_given_tag(word, tag, train_bag = train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)#total number of times the passed tag occurred in train_bag
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
#now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)


    return (count_w_given_tag, count_tag)


# In[114]:


def t2_given_t1(t2, t1, train_bag = train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


# In[115]:


tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)): 
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]

print(tags_matrix)


# In[116]:


tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))
display(tags_df)


# In[117]:


def Viterbi(words, train_bag = train_tagged_words):
    tags = []
    T = list(set([pair[1] for pair in train_bag]))

    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[tags[-1], tag]

            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]
            tags_probability = emission_p * transition_p    
            p.append(tags_probability)

        pmax = max(p)
        # getting state for which probability is maximum
        tags_max = T[p.index(pmax)] 
        tags.append(tags_max)
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

# In[118]:


random.seed(1234)  
rndom = [random.randint(1,len(test_set)) for x in range(10)]
test_run = [test_set[i] for i in rndom]
test_run_base = [tup for sent in test_run for tup in sent]
test_tagged_words = [tup[0] for sent in test_run for tup in sent]


# In[119]:


start = time.time()
tagged_seq = Viterbi(test_tagged_words)
end = time.time()
difference = end-start

print("Time taken in seconds: ", difference)

# accuracy
check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 

accuracy = len(check)/len(tagged_seq)
print('Viterbi Algorithm Accuracy: ',accuracy*100)


# In[ ]:





# In[ ]:


#Number 1


# In[24]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
dfi = pd.read_csv (r'D:\BRAC\CSE712\assignment\imdb_440\IMDB Dataset.csv')
df = dfi.sample(n=1000)  #frac = 0.5
df['review'] = df['review'].str.lower()
df['review']


# In[25]:


import numpy as np
docs = np.array(df['review'].tolist())  # .features, dtype=np.str


# In[26]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
tf = vect.fit_transform(docs).toarray()
tf


# In[27]:


# manual
n_docs = len(docs)
df = np.sum(tf != 0, axis = 0)
idf = np.log((1 + n_docs) / (1 + df)) + 1
tf_idf = tf[0] * idf
tf_norm = tf_idf / np.sqrt(np.sum(tf_idf ** 2))
print(tf_norm)
print()
print(tf_idf)
print()



# In[48]:


A = vect.vocabulary_
newA =sorted(A, key=A.get, reverse=True)[:5]
newA


# In[ ]:





# In[84]:


def cosine_similarity(A, B):

    dot = np.dot(A,B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot/(norma*normb)

    ### END CODE HERE ###
    return cos


# In[102]:


import pickle
import numpy as np
import pandas as pd

#from utils import get_vectors
import nltk
from gensim.models import KeyedVectors


embeddings= KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary = True)
f = open('D:\BRAC\CSE712\glove.6B.100d.txt', 'rb').read()
set_words = set(nltk.word_tokenize(f))
select_words = words = ['king', 'Queen', 'Cat', 'Dog', 'Man', 'Woman', 'city', 'town', 'village', 'country', 'continent', 'petroleum', 'joyful']
for w in select_words:
    set_words.add(w)

def get_word_embeddings(embeddings):

    word_embeddings = {}
    for word in embeddings.vocab:
        if word in set_words:
            word_embeddings[word] = embeddings[word]
    return word_embeddings


# Testing your function
word_embeddings = get_word_embeddings(embeddings)
print(len(word_embeddings))
pickle.dump( word_embeddings, open( "word_embeddings_subset.p", "wb" ) )
6+4


# In[103]:


king = word_embeddings['king']
queen = word_embeddings['queen']

cosine_similarity(king, queen)


# In[104]:


Man = word_embeddings['Man']
Woman = word_embeddings['Woman']

cosine_similarity(Man, Woman)


# In[105]:


Cat = word_embeddings['Cat']
Dog = word_embeddings['Dog']

cosine_similarity(Cat, Dog)


# In[ ]:
