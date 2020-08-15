#!/usr/bin/python
# -*- coding: utf-8 -*-

from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from joblib import dump, load
import pickle

# load the model from disk

filename = 'nlp_model_new.pkl'
cv = pickle.load(open('transform_new.pkl', 'rb'))
tfidf = pickle.load(open('transform.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data)
        vectfinal = tfidf.transform(vect)
        my_prediction = model.predict(vectfinal)
    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)


			