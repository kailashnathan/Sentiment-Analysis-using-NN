# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:47:14 2017

@author: Neelesh
"""

import pandas as pd

from beautiful_data import get_data




data_size = 50000
max_features = 200
# No. of words to be used as features. Any more : Truncate ; Any less: Pad


print("\nFirst let's load Data: ")

data = get_data(data_size, max_features)

print("\nOur (beautiful) data looks something like this: \n", data.head)

print("\nSaving dataet to Cleaned",data_size,".csv:" )
data.to_csv("Balanced_Cleaned_data"+str(data_size)+".csv")


print("\nNow let's do something with it.")

