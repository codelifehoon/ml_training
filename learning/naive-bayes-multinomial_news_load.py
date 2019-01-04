from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import  sem
from sklearn import  metrics
import  pprint as pp
import pandas as pd
from sklearn.externals import joblib




filename = 'naive-bayes-multinomial_news.joblib.pkl'
clf = joblib.load(filename)

my_data = ['관해서 형태소분석을 실시'
    ,'텍스트중에서 나온 활용어'
    ,'I am sure some bashers of Pens fans are pretty confused about the lack of any kind of posts about the recent Pens massacre of the Devils. Actually,'
           ]

predict = clf.predict(my_data)
predict_proba = clf.predict_proba(my_data)
predict_data_max = []
print(predict)
print(predict_proba)
print( len(clf.named_steps['vect'].get_feature_names()))
print( clf.named_steps['vect'].get_feature_names())


for data in predict_proba:
    predict_data_max.append(max(data))


predict_data = pd.DataFrame({'predict':predict
                                ,'maxval':predict_data_max
                                ,'validation':1})

predict_data.loc[predict_data['maxval']<0.95,'validation':'validation'] =  0
print(predict_data)


predict_data = predict_data.loc[predict_data['maxval']>0.95,:]
print(predict_data)





