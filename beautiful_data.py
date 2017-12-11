# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 01:29:42 2017

@author: Neelesh
"""


import pandas as pd
import balance as b



def get_data(data_size, max_features):

#    data_size = int(input("\n***ENTER DATASET SIZE***: "))
    print("\nLoading ",data_size," records.......")
    
    #1. First load all the data (EnglishReviewsClean.csv) : Self-explanatory name
    dataset = pd.read_csv('C:/Users/Neelesh/Desktop/Yelp Sentiment analysis/Data preprocessing/EnglishReviewsClean.csv')
    dataset.drop('Unnamed: 0', axis=1, inplace=True)
    print("\nBasic English Reviews data:\n",dataset.head(5))
    print("\nDataset Size: ",dataset.count())
    
    #2. Extract required data
    print("\nExtract ", data_size, "rows\n")
    dataset = dataset[:data_size]
    
    #3. Balance and Polarize the dataset
    print("\nBalance and Polarize the dataset:")
    dataset = b.balance_and_polarize(dataset)
    print('\nBalanced reviews',dataset.head())
    dataset.to_csv("BalancedReviews"+str(data_size)+".csv")
    
    
    #4. Clean the review text data
    print("\nCleaning the review text data.")
    dataset['text'] = dataset.text.apply(b.clean)
    print('\nDataset with reviews cleaned:\n',dataset.head)
    dataset.to_csv("CleanedReviews"+str(data_size)+".csv")
    
    #5. Check Text lengths
    print("\nCheck Text lengths: ")
    b.text_lengths(dataset)
    
    return dataset