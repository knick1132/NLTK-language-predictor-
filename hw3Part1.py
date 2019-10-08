
# coding: utf-8

# In[1]:


import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string
import re
from nltk.stem import PorterStemmer
from nltk.util import ngrams
import pickle


# In[2]:



def prog1(filename):#create unigrams and bigrams of the training files
    file = open(filename, encoding="utf8")
    text = file.read()
    file.close() #close file

    text = text.replace('\n',' ')
    tokens = nltk.word_tokenize(text) #tokenize 

    unigrams = ngrams(tokens, 1)

    unigram_dict = {}#create unigrame
    for unigram in set(unigrams):
        unigram_dict[unigram[0]] = text.count(unigram[0])


    bigrams = ngrams(tokens, 2)

    bigram_dict = {}#create bigram
    for bigram in set(bigrams):
        if bigram not in bigram_dict:
            bi = bigram[0] + ' ' + bigram[1]
            bigram_dict[bi] = text.count(bi)

    return unigram_dict, bigram_dict


# In[3]:


#"LangId.train.English"


# In[4]:


if __name__ == "__main__":
    English_unigram, English_bigram = prog1("LangId.train.English")
    French_unigram, French_bigram = prog1("LangId.train.French")
    Italian_unigram, Italian_bigram = prog1("LangId.train.Italian")


# In[5]:


with open('English_unigram', 'wb') as handle:#pickle the training files's unigrams and bigrams for part 2
    pickle.dump(English_unigram, handle)
with open('English_bigram', 'wb') as handle:
    pickle.dump(English_bigram, handle)

    
with open('French_unigram', 'wb') as handle:
    pickle.dump(French_unigram, handle)
with open('French_bigram', 'wb') as handle:
    pickle.dump(French_bigram, handle)
    
    
with open('Italian_unigram', 'wb') as handle:
    pickle.dump(Italian_unigram, handle)
with open('Italian_bigram', 'wb') as handle:
    pickle.dump(Italian_bigram, handle)  

#with open('example.pickle', 'rb') as handle:
#   new_dict = pickle.load(handle)

