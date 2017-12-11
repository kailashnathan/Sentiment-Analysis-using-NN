# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 03:12:17 2017

@author: Neelesh
"""

import numpy as np

from beautiful_data import get_data

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from collections import defaultdict


print ("\nBASELINE MODELS:")
accuracies = defaultdict()

##### INPUT THE VARIABLES #####

data_size = 50000
max_features = 200
# No. of words to be used as features. Any more : Truncate ; Any less: Pad


print("\nFirst let's load Data: ")

data = get_data(data_size, max_features)

print("\nOur (beautiful) data looks something like this: \n", data.head)

print("\nNow let's do something with it.")



# Extract data(text) and labels(sentiment)
reviews=data['text']
sentiment=data['sentiment']

tokenizer = Tokenizer(num_words=max_features+1, 
                      split=' ',
                      lower=True)
tokenizer.fit_on_texts(reviews.values)


X = tokenizer.texts_to_sequences(reviews.values)


X = pad_sequences(X)
Y = sentiment
print("\nData shape: ",X.shape,"\nLabels shape: ",Y.shape,"\nNo. of features: ",X.shape[1],"\nMax features: ",max_features)

# the data, shuffled and split between train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=True, random_state=1)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print('\nTrain data shape: ', x_train.shape)
print('\nTest data shape: ', x_test.shape)

print("\nTrain labels shape: ", y_train.shape)
print("\nTest labels shape: ",y_test.shape)

print("\nSample train: ",x_train[0,:])
print("\nSample test: ",x_test[0,:])




# 1. Model Specification: Most Common Class Baseline

most_common = data['sentiment'].value_counts()
tot = len(data['sentiment'])
mcc_acc = most_common/tot
print("\nMCC Baseline accuracy: ", mcc_acc)
accuracies['MCC'] = mcc_acc

# 2. Model Specification: Multinomial Naive Bayes
nb = MultinomialNB()
nb.fit(x_train, y_train)

preds = nb.predict(x_test)

mnb_acc = accuracy_score(y_test,preds)
print("\nMultinomial NB Baseline accuracy: ", mnb_acc)
accuracies['MultinomialNB'] = mnb_acc

# 3. Model Specification: Random Forest
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(x_train, y_train)
preds = forest.predict(x_test)
rf_acc = accuracy_score(y_test,preds)
print("\nRandom Forest accuracy: ",rf_acc)
accuracies['RandomForest'] = rf_acc

# 4. Model Specification: Perceptron
p = Perceptron()
p=p.fit(x_train, y_train)
preds=p.predict(x_test)
perc_acc=accuracy_score(y_test,preds)
print("\nPerceptron accuracy: ", perc_acc)
accuracies['Perceptron'] = perc_acc


print("\nBaseline accuracies: ", accuracies)

