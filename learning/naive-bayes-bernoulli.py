from sklearn.naive_bayes import BernoulliNB
import numpy as np
np.random.seed(0)

X = np.array([
    [0, 1, 1, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [0, 1, 1, 0]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


model_bern = BernoulliNB().fit(X, y)

fc = model_bern.feature_count_

# smoothing(avoid Vanishing-Gradient-Problem )
theta = np.exp(model_bern.feature_log_prob_)

print(model_bern.classes_)
print(model_bern.class_count_)
print(fc)
print(theta)
print(model_bern.alpha)


x_new = np.array([1, 1, 0, 0])
print(model_bern.predict_proba([x_new]))

