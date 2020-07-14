# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:42:22 2020

@author: Rupesh
"""


import pandas as pd
import numpy as np
import pickle as pickle


dataset=pd.read_csv("spam.csv", encoding="latin-1")


corpus=[]
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

for i in range(len(dataset)):
    review=dataset["message"][i]
    filtered_review=re.sub("[^a-zA-Z]"," ",review).lower().split()
    filtered_review=[ps.stem(words) for words in filtered_review if not words in set(stopwords.words("english"))]
    filtered_review= " ".join(filtered_review)
    corpus.append(filtered_review)



#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=4000)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,0]

pickle.dump(cv, open('tranform.pickle', 'wb'))


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)



from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)

from sklearn.model_selection import  cross_val_score
score=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracy=score.mean()

pickle.dump(classifier,open("nlp_model.pickle",'wb'))
    