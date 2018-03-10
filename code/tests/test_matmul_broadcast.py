import numpy as np
import tensorflow as tf
H = 2
N = 4
M = 3
BS = 5

if __name__== "__main__":
    w_1 = np.array([1., 2., 3., 4.])
    c = np.array([[[1., 2., 3., 4.], [5., 6., 7., 8.]]]) # BS x N x 2H
    c = np.tile(c, [BS, 1, 1])

    contexts = tf.get_variable('contexts', initializer=c)
    contexts = tf.reshape(contexts, (-1, 4))
    weights = tf.get_variable('weights', initializer=w_1)
    weights = tf.reshape(weights, (4, 1))
    multiply = tf.matmul(contexts, weights)
    multiply = tf.reshape(multiply, (-1, 2, 1))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        weights_results = sess.run(multiply)
        print(weights_results)
