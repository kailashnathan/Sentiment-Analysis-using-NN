# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 18:42:43 2017

@author: Kailash Nathan
"""
import pandas as pd
import numpy as np

#len(toy)
#print (toy[:200])
datas = pd.read_csv("cleandatatoy.csv",)
datas=datas[:1000]
#print (datas.values)
datas.head
'''
#print (datas.values)
#data['text']
data=datas[['text','stars']]
print (data)
data['sentiment'] = np.where(datas['stars']>=4, 'Positive', 'Negative')
data=data[['text','sentiment']]

d=datas[:1000]
'''
pos=datas.loc[datas['sentiment']==1]
len(pos)
neg=datas.loc[datas['sentiment']==0]
len(neg)

pos=pos.drop('stars',1)
neg=neg.drop('stars',1)


pos.to_csv('kaggledatatoypositivess.txt', header=True, index=False, sep='\t', mode='a')

neg.to_csv('kaggledatatoynegativess.txt', header=True, index=False, sep='\t', mode='a')

pos22=[]
neg22=[]

pos22 = list(open('kaggledatatoypositives.txt', "r").readlines())
neg22 = list(open('kaggledatatoynegatives.txt', "r").readlines())
print (len(neg22),len(pos22))
#print(neg22)