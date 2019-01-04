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



def evaluate_cross_validation(clf, X, y, K):
    # cv = KFold(len(y),K,shuffle=True,random_state=0)
    scores = cross_val_score(clf,X,y,cv=K)
    print('score:',scores)
    print('mean score:',np.mean(scores) , 'sem score:',sem(scores))

def train_and_evaluate(clf,X_train,X_test,y_train,y_test):

    clf.fit(X_train,y_train)
    print('Accurancy on training set:')
    print(clf.score(X_train,y_train))
    print('Accurancy on test set:')
    print(clf.score(X_test,y_test))

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)


    print('predict_proba Repost:')
    print(y_pred_proba)

    print('Classification Repost:')
    pp.pprint(metrics.classification_report(y_test,y_pred))
    print('Classification Matrix:')
    pp.pprint(metrics.confusion_matrix(y_test,y_pred))


news = fetch_20newsgroups(subset="all")

X = news.data
y = news.target

# filename = news.filenames
# print(filename[0:10])
# print(news.keys())
# print(news.description)
# print(type(X),type(y))
# print(news.target_names)

split_rate = 0.75
split_size = int(len(X) * split_rate)
X_train,X_test = X[:split_size] , X[split_size:]
y_train,y_test = y[:split_size] , y[split_size:]
my_data = ['관해서 형태소분석을 실시','텍스트중에서 나온 활용어']

'''
clf_1 = Pipeline([
                    ('vect',CountVectorizer(stop_words="english",token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b"))
                    ,('clf',MultinomialNB(alpha=0.01))
                ])

clf_2 = Pipeline([
    ('vect',TfidfVectorizer())
    ,('clf',MultinomialNB())
])
'''
clf_3 = Pipeline([
    ('vect',TfidfVectorizer(stop_words="english",token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b"))
    ,('clf',MultinomialNB(alpha=0.01))
])


# clfs = [clf_1,clf_2,clf_3]
clfs = [clf_3]

#
# for clf in clfs:
#     evaluate_cross_validation(clf, X_train, y_train, 5)

train_and_evaluate(clf_3,X_train,X_test,y_train,y_test)

clf = clf_3

y_predict = clf.predict(my_data)
y_predict_proba = clf.predict_proba(my_data)
print(y_predict)
print(y_predict_proba)
print( len(clf_3.named_steps['vect'].get_feature_names()))
print( clf_3.named_steps['vect'].get_feature_names())

