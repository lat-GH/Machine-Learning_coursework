# Step 1: Load all the components we need to build this classifier

import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

# Step 2: Read the data file we have. It consists of two fields

tweets = pd.read_csv('C:\Users\latma\OneDrive\Documents\ComputerScience_Yr2\Machine_Learning\SentimentAnalysis_python\car_reviews.csv', header = 0, encoding = 'latin-1')

#tweets[:3]

# The tweets could do with some clean up such as removing punctuation, capitalisation, stop words. And stemming or lemmatisation.