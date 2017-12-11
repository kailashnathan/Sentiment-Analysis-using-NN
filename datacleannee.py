# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:04:39 2017

@author: Kailash Nathan
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 00:56:47 2017

@author: Neelesh
"""
# Import all that you need for preprocessing and cleaning the data

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import string
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# First load all the data (EnglishReviewsClean.csv) : Self-explanatory name
dataset = pd.read_csv('ReviewsClean.csv')
dataset.drop('Unnamed: 0', axis=1, inplace=True)
print('\nBasic English Reviews data:\n',dataset.head())
#print(dataset.describe())

# Check the class-wise record-counts
print(dataset['stars'].value_counts())

# Find the minority class to guide your balancing
minor_count = dataset.groupby(['stars']).apply(lambda x: x.shape[0]).min()
print(minor_count)
# You'll find that the minority class has around 3.9 lac records

# Use sampling to sample 3.75 lac records in each class: Inherently random, doesn't hurt
print("\nBalancing the dataset...")
dataset = dataset.groupby(['stars']).apply(
        lambda x: x.sample(375000) #, random_state=7)
    ).drop(['stars'], axis=1).reset_index().set_index('level_1')

dataset.sort_index(inplace=True)

# Now, the dataset size is 1.5 million records and each class has 3.75 lac recs
print("\nBalanced dataset:\n")
print(dataset.head)
#print(dataset.describe())

# Sanity check the class sizes
print(dataset['stars'].value_counts())

# Make the class distribution binary : Positive & Negative sentiment
# Introduce a column 'sentiment' whose value is 1 (Positive ) for stars>=4, 0 (Negative) otherwise
print("\nBinarize classes into negative(0) and positive(1):\n")
dataset['sentiment'] = np.where(dataset['stars']>=4, 1, 0)
print(dataset[ dataset['sentiment'] == 1].size)
print(dataset[ dataset['sentiment'] == 0  ].size)

# Sanity check sentiment against stars
print('\n',dataset.head)
#print(dataset.describe())

# Now, our 'functional' dataset will consist of text and sentiment columns
dataset=dataset[['text','sentiment']]


# Clean the review text data
dataset['text'] = dataset['text'].apply(lambda x: x.lower())
dataset['text'] = dataset['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))




# Final sanity check!! 
print('\n',dataset.head)
data=dataset[:1500]
data.to_csv('cleandatatoy.csv',header=True, index=False, sep='\t', mode='a')
print(data.head)

pos=data.loc[data['sentiment']==1]
len(pos)
neg=data.loc[data['sentiment']==0]
len(neg)
pos.to_csv('kaggledatatoypositiveswc.txt', header=True, index=False, sep='\t', mode='a')

neg.to_csv('kaggledatatoynegativeswc.txt', header=True, index=False, sep='\t', mode='a')

pos22=[]
neg22=[]

pos22 = list(open('kaggledatatoypositiveswc.txt', "r").readlines())
neg22 = list(open('kaggledatatoynegativeswc.txt', "r").readlines())
print (len(neg22),len(pos22))
'''

# Add a column called text length
dataset['text length'] = dataset['text'].apply(len)
print(dataset.head())

'''