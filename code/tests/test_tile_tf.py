import numpy as np
import tensorflow as tf
H = 2
N = 2
M = 3
BS = 5

def my_softmax(arr):
    max_elements = np.reshape(np.max(arr, axis = 2), (BS, N, 1))
    arr = arr - max_elements
    exp_array = np.exp(arr)
    print (exp_array)
    sum_array = np.reshape(np.sum(exp_array, axis=2), (BS, N, 1))
    return exp_array /sum_array

def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float64')) * (-1e30) # -large where there's padding, 0 elsewhere
    print (exp_mask)
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist


def test_build_similarity(contexts, questions):
    w_sim_1 = tf.get_variable('w_sim_1',
        initializer=w_1) # 2 * H
    w_sim_2 = tf.get_variable('w_sim_2',
        initializer=w_2) # 2 * self.hidden_size
    w_sim_3 = tf.get_variable('w_sim_3',
        initializer=w_3) # 2 * self.hidden_size
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
    return S

def test_build_sim_mask():
    context_mask = np.array([True, True]) # BS x N
    question_mask = np.array([True, True, False]) # BS x M
    context_mask = np.tile(context_mask, [BS, 1])
    question_mask = np.tile(question_mask, [BS, 1])
    context_mask = tf.get_variable('context_mask', initializer=context_mask)
    question_mask = tf.get_variable('question_mask', initializer=question_mask)
    context_mask = tf.expand_dims(context_mask, -1) # BS x N x 1
    question_mask = tf.expand_dims(question_mask, -1) # BS x M x 1
    question_mask = tf.transpose(question_mask, (0, 2, 1)) # BS x 1 x M
    sim_mask = tf.matmul(tf.cast(context_mask, dtype=tf.int32),
            tf.cast(question_mask, dtype=tf.int32)) # BS x N x M
    return sim_mask

def test_build_c2q(S, S_mask, questions):
    _, alpha = masked_softmax(S, mask, 2) # BS x N x M
    return tf.matmul(alpha, questions)

def test_build_q2c(S, S_mask, contexts):
    # S = BS x N x M
    # contexts = BS x N x 2H
    m = tf.reduce_max(S * tf.cast(S_mask, dtype=tf.float64), axis=2) # BS x N
    beta = tf.expand_dims(tf.nn.softmax(m), -1) # BS x N x 1
    beta = tf.transpose(beta, (0, 2, 1))
    q2c = tf.matmul(beta, contexts)
    return m, beta, q2c

def test_concatenation(c2q, q2c):
    q2c = tf.tile(q2c, (1, N, 1))
    output = tf.concat([c2q, q2c], axis=2)
    tf.assert_equal(tf.shape(output), [BS, N, 4*H])
    return output

if __name__== "__main__":
    w_1 = np.array([1., 2., 3., 4.])
    w_2 = np.array([5., 6., 7., 8.])
    w_3 = np.array([13., 12., 11., 10.])

    c = np.array([[[1., 2., 3., 4.], [5., 6., 7., 8.]]]) # BS x N x 2H
    q = np.array([[[1., 2., 3., 0.], [5., 6., 7., 4.], [8., 9. , 10., 11.]]]) # BS x M x 2H
    c = np.tile(c, [BS, 1, 1])
    q = np.tile(q, [BS, 1, 1])


    questions = tf.get_variable('questions', initializer=q)
    contexts = tf.get_variable('contexts', initializer=c)

    S = test_build_similarity(contexts, questions)
    mask = test_build_sim_mask()
    c2q = test_build_c2q(S, mask, questions)
    m, beta, q2c = test_build_q2c(S, mask, contexts)
    output = test_concatenation(c2q, q2c)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        S_result, mask_result, c2q_r = sess.run([S, mask, c2q])
        actual_result = np.tile(np.array([[228, 772, 1372], [548, 1828, 3140]]), [BS, 1, 1])
        assert np.array_equal(actual_result, S_result), 'Arrays are not equal'
        print ("Building similarity matrix is successful!")
        print ("Context 2 Question attention")
        m_r, beta_r, q2c_r = sess.run([m, beta, q2c])
        output_r = sess.run(output)
