# -*- coding: utf-8 -*-
"""embeddings_chain_mulit_outputs_choice1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Xg9Qc1DDFGybF1ssSwK2V-XL3QK4CIZl
"""

import numpy as np
import pandas as pd
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

def get_tfidf_embd_type2(df:pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    data = 0
    #data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    data = df['Type 2']
    X = tfidfconverter.fit_transform(data).toarray()
    return X

def get_tfidf_embd_type2_3(df:pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    #data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    data = 0
    for t2 in range(len(df)):
      if df['Type 2'].iloc[t2] == df['y2'].iloc[t2]:
        data += df['Type 2'].iloc[t2] + df['Type 3'].iloc[t2] 
      elif df['Type 2'].iloc[t2] != df['y2'].iloc[t2]:
        data += null
    X = tfidfconverter.fit_transform(data).toarray()
    return X

def get_tfidf_embd_type2_3_4(df:pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    #data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    data = 0
    for t2 in range(len(df)):
      if df['Type 2'].iloc[t2] == df['y2'].iloc[t2]:
        data += df['Type 2'].iloc[t2] + df['Type 3'].iloc[t2] + df['Type 4'].iloc[t2] 
      elif df['Type 2'].iloc[t2] != df['y2'].iloc[t2]:
        data += null
    X = tfidfconverter.fit_transform(data).toarray()
    return X

def combine_embd(X1, X2):
    return np.concatenate((X1, X2), axis=1)