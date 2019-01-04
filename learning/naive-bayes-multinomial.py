from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
np.random.seed(0)

X = np.array([
    [3, 4, 1, 2],
    [3, 5, 1, 1],
    [3, 3, 0, 4],
    [3, 4, 1, 2],
    [1, 2, 1, 4],
    [0, 0, 5, 3],
    [1, 2, 4, 1],
    [1, 1, 4, 2],
    [0, 1, 2, 5],
    [2, 1, 2, 3]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


model_bern = MultinomialNB().fit(X, y)

fc = model_bern.feature_count_
# smothing(avoid Vanishing-Gradient-Problem )
theta = np.exp(model_bern.feature_log_prob_)

# print(model_bern.classes_)
# print(model_bern.class_count_)
# print(fc)
# print(theta)
# print(model_bern.alpha)


x_new = np.array([[3, 4, 1, 0],[9, 9, 9, 9]])
print(model_bern.predict(x_new))
print(model_bern.predict_proba(x_new))

predict = model_bern.predict(x_new)
predict_proba = model_bern.predict_proba(x_new)
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








