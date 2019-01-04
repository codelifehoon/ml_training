# https://www.tensorflow.org/api_guides/python/test

import tensorflow as tf


x_date = [[1,3]
          ,[2,4]
          ,[10,20]
            ]

w_date = [[1,0]
    ,[0,1]
    ]


X = tf.placeholder("float",[None,2])
W = tf.placeholder("float",[None,2])

R1 = tf.reduce_sum(x_date)
R2 = tf.reduce_sum(x_date,0)
R3 = tf.reduce_sum(x_date,1)


matmul = tf.matmul(X,W)
hypothesis = tf.reduce_sum(matmul,1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    result = sess.run(R1,feed_dict={X: x_date})
    print(result)

    result = sess.run(R2,feed_dict={X: x_date})
    print(result)

    result = sess.run(R3,feed_dict={X: x_date})
    print(result)

    result = sess.run(matmul,feed_dict={X: x_date,W: w_date})
    print(result)

    result = sess.run(hypothesis,feed_dict={X: x_date,W: w_date})
    print(result)



