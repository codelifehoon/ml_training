import sys
sys.path.insert(0, '/Users/codelife/Developer/tensorFlow/DeepLearningZeroToAll')

from gensim.models import word2vec
import tensorflow as tf
import numpy as np
import printutils as pu
import matplotlib.pyplot as plt


sess = tf.Session()
sess.run(tf.global_variables_initializer())

a = np.zeros((2,2))
print(a)
# 출력:
# [[ 0.  0.]
#  [ 0.  0.]]

a = np.ones((2,3))
print(a)
# 출력:
# [[ 1.  1.  1.]
#  [ 1.  1.  1.]]

a = np.full((2,3), 5)
print(a)
# 출력:
# [[5 5 5]
#  [5 5 5]]

a = np.eye(3)
print(a)
# 출력:
# [[ 1.  0.  0.]
#  [ 0.  1.  0.]
#  [ 0.  0.  1.]]

a = np.array(range(20)).reshape((5,2,2))
print(a)
# 출력:
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]
#  [15 16 17 18 19]]

img9x9 = np.array([1,2,3,4,5,6,7,8,9])
img9x9 = sess.run(tf.reshape(img9x9,[3,3]))
print(img9x9)

img9x9_2 = np.array([1,2,3,4,5,6,7,8,9],dtype=np.float32).reshape(3,3)
print(img9x9_2)


plt.imshow( img9x9_2, cmap='Greys')
plt.show()

