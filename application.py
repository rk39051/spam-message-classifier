# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 00:52:02 2020

@author: Rupesh
"""


from flask import render_template,request,url_for,Flask
import  pandas as pd
import numpy as np
import nltk
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# load the model from disk
classifier = pickle.load(open('nlp_model.pickle', 'rb'))
cv=pickle.load(open('tranform.pickle','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/prediction',methods=['POST'])
def prediction():
	if request.method == 'POST':
		message = request.form['text']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = classifier.predict(vect)
	return render_template('prediction.html',prediction = my_prediction)



if __name__ == '__main__':
	#app.run(debug=True)
    app.run(host="0.0.0.0",port=8080)
