from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold


import numpy as np
import pandas as pd
import  matplotlib
matplotlib.use('TkAgg')

news = fetch_20newsgroups(subset="test")
X = news.data
y = news.target



model1 = Pipeline([
    ('vect', CountVectorizer()),
    ('model', MultinomialNB()),
])
# model2 = Pipeline([
#     ('vect', TfidfVectorizer()),
#     ('model', MultinomialNB()),
# ])
# model3 = Pipeline([
#     ('vect', TfidfVectorizer(stop_words="english")),
#     ('model', MultinomialNB()),
# ])
# model4 = Pipeline([
#     ('vect', TfidfVectorizer(stop_words="english",
#                              token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b")),
#     ('model', MultinomialNB()),
# ])


for i, model in enumerate([model1]):
    scores = cross_val_score(model, X, y, cv=5)
    model.fit(X,y)
    print(("Model{0:d}: Mean score: {1:.3f}").format(i, np.mean(scores)))


predict_data = np.array(['I am sure some bashers'
                            ,'am a little confused on all of the models of the 88-89 bonnevilles.'
                         ])


predict = model1.predict(predict_data)
predict_proba = model1.predict_proba(predict_data)
predict_data_max = []

for data in predict_proba:
    predict_data_max.append(max(data))


predict_data = pd.DataFrame({'predict':predict
                                ,'maxval':predict_data_max
                                ,'validation':1})

predict_data.loc[predict_data['maxval']<0.95,'validation':'validation'] =  0
print(predict_data)


predict_data = predict_data.loc[predict_data['maxval']>0.95,:]
print(predict_data)
