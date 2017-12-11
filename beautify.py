# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 01:29:42 2017

@author: Neelesh
"""


import pandas as pd
import balance as b
import re

#import numpy as np





# First load all the data (EnglishReviewsClean.csv) : Self-explanatory name
dataset = pd.read_csv('C:/Users/Neelesh/Desktop/Yelp Sentiment analysis/Data preprocessing/EnglishReviewsClean.csv')
dataset.drop('Unnamed: 0', axis=1, inplace=True)
print('\nBasic English Reviews data:\n',dataset.head())

print("\nCut the dataset for dev..")
dataset = dataset[:10000]

print("\nBalance and Polarize the dataset\n")
dataset = b.balance_and_polarize(dataset)
print('\nBalanced reviews',dataset.head())
dataset.to_csv("BalancedReviews.csv")


# Clean the review text data
print("\nCleaning the review text data.")
dataset['text'] = dataset.text.apply(b.clean) #(dataset.text), axis=1) #, axis=1)

print('\nDataset with reviews cleaned:\n',dataset.head)

dataset['text'] = dataset['text'].apply((lambda x: re.sub('[^a-zA-Z0-9\s]','',x)))

print('\nDataset with reviews cleaned:\n',dataset.head)