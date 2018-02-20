import numpy as np
import tensorflow as tf


H = 2
N = 2
M = 3
BS = 10

w_1 = np.array([1., 2., 3., 4.])
w_2 = np.array([5., 6., 7., 8.])
w_3 = np.array([13., 12., 11., 10.])

c = np.array([[[1., 2., 3., 4.], [5., 6., 7., 8.]]]) # BS x N x 2H
q = np.array([[[1., 2., 3., 0.], [5., 6., 7., 4.], [8., 9. , 10., 11.]]]) # BS x M x 2H
c = np.tile(c, [BS, 1, 1])
q = np.tile(q, [BS, 1, 1])

w_sim_1 = tf.get_variable('w_sim_1',
    initializer=w_1) # 2 * H
w_sim_2 = tf.get_variable('w_sim_2',
    initializer=w_2) # 2 * self.hidden_size
w_sim_3 = tf.get_variable('w_sim_3',
    initializer=w_3) # 2 * self.hidden_size
questions = tf.get_variable('questions', initializer=q)
contexts = tf.get_variable('contexts', initializer=c)


q_tile = tf.tile(tf.expand_dims(questions, 0), [N, 1, 1, 1]) #  N x BS x M x 2H
q_tile = tf.transpose(q_tile, (1, 0, 3, 2)) # BS x N x 2H x M

contexts = tf.expand_dims(contexts, -1) # BS x N x 2H x 1

result = (contexts * q_tile) # BS x N x 2H x M

tf.assert_equal(tf.shape(result), [BS, N, 2 * H, M])
result = tf.transpose(result, (0, 1, 3, 2)) # BS x N x M x 2H
result = tf.reshape(result, (-1, N * M, 2 * H)) # BS x (NxM) x 2H
tf.assert_equal(tf.shape(result), [BS, N*M, 2*H])

w_sim_1 = tf.tile(tf.expand_dims(w_sim_1, 0), [BS, 1])
w_sim_2 = tf.tile(tf.expand_dims(w_sim_2, 0), [BS, 1])
w_sim_3 = tf.tile(tf.expand_dims(w_sim_3, 0), [BS, 1])
term1 = tf.matmul(tf.reshape(contexts, (BS, N, 2*H)), tf.expand_dims(w_sim_1, -1)) # BS x N
term2 = tf.matmul(questions, tf.expand_dims(w_sim_2, -1)) # BS x M
term3 = tf.matmul(result, tf.expand_dims(w_sim_3, -1)) # BS x N x M
term3 = tf.reshape(term3, (BS, N, M))
S = tf.reshape(term1,(-1, N, 1)) + term3 + tf.reshape(term2, (-1, 1, M))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    S_result = sess.run(S)
    actual_result = np.tile(np.array([[228, 772, 1372], [548, 1828, 3140]]), [BS, 1, 1])
    assert np.array_equal(actual_result, S_result), 'Arrays are not equal'
    print ("Test is successful!")
