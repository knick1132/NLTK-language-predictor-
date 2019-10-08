
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


def compute_prob(text, unigram_dict, bigram_dict, v):#compute probablity using laplace
    
    text = text.replace('\n',' ')
    text = nltk.word_tokenize(text)
    
    unigrams_test = ngrams(text, 1)
    bigrams_test = ngrams(text, 2)
 
    p_laplace = 1
    b = 0
    u = 0
    for bigram in bigrams_test:
        
        temp = bigram[0] + " " + bigram[1]
        
        b = b + bigram_dict[temp] if temp in bigram_dict else 0
        u = u + unigram_dict[bigram[0]] if bigram[0] in unigram_dict else 0
        p_laplace = p_laplace * ((b + 1) / (u + v))

    return p_laplace
    



with open('English_unigram', 'rb') as handle:#read pickled dictionaries from part 1
    English_unigram = pickle.load(handle)
with open('English_bigram', 'rb') as handle:
    English_bigram = pickle.load(handle)
    
with open('French_unigram', 'rb') as handle:
    French_unigram = pickle.load(handle)
with open('French_bigram', 'rb') as handle:
    French_bigram = pickle.load(handle)
    
with open('Italian_unigram', 'rb') as handle:
    Italian_unigram = pickle.load(handle)
with open('Italian_bigram', 'rb') as handle:
    Italian_bigram = pickle.load(handle)
    
file = open("LangId.test")#read testing file
text = file.read()
text = text.split('\n')
file.close() #close file

v = len(English_unigram) + len(French_unigram) + len( Italian_unigram)#calculate v

count = 0

text.pop()

file = open("program2A.txt","w+")

for line in text:#loop that computes the greatest probablity langauge
    count += 1
    tmp = []
    #print(line)
    english = compute_prob(line, English_unigram, English_bigram, v)
    french = compute_prob(line, French_unigram, French_bigram, v)
    italian = compute_prob(line, Italian_unigram, Italian_bigram, v)
    
    tmp.append(english)
    tmp.append(french)
    tmp.append(italian)
    
    maxi = max(tmp)#gets the highest probablity language
    if maxi == english: 
        file.write("%d English\n" % count)
    else: 
        if maxi == french: 
            file.write("%d French\n" % count)
        else:
            if maxi == italian: 
                file.write("%d Italian\n" % count)
    
    
file.close()

    


# In[3]:


file = open("program2A.txt")#create a file for part 2b
test = file.read()
test = test.split('\n')
test.pop()
file.close() #close file


file = open("LangId.SOL") #open SOL for accuarcy comparison
sol = file.read()
sol = sol.split('\n')
sol.pop()
file.close() #close file


# In[6]:


count = 0

for key in sol:#get how many of the classifications are correct
    if key in test:
        count +=1
#print accuarcy        
print("overall accuracy: ",  str(round(count/len(sol), 2)), '\n================================')

print("\n\nincorrect lines\n=====================================")#print accuarcy and incorrect lines 
matching = 0
count = 0

for key in sol:
    count += 1
    if key in test:
        matching +=1
    else:
        print("line: ", key.split(' ')[0], '\n', "accuracy: ",  str(round(matching/count, 2)))

