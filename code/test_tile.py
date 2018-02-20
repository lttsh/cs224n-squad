import numpy as np

N = 2
M = 3
H = 2

c = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]) # N x 2H
q = np.array([[1, 2, 3, 0], [5, 6, 7, 4], [8, 9 , 10, 11]]) # M x 2H

q_tile = np.tile(q, (N, 1, 1)) # N x M x 2H
q_tile = np.transpose(q_tile, (0, 2, 1)) # N x 2H x M
c = np.reshape(c, (N, 2*H, 1))

result = (c * q_tile) # N x 2H x M
result = np.transpose(result, (0, 2, 1))
result = np.reshape(result, (-1, 2 * H ))
print (result) # C_1 * Q ; C_2 * Q

w_1 = np.array([1, 2, 3, 4])
w_2 = np.array([5, 6, 7, 8])
w_3 = np.array([13, 12, 11, 10])


# term1
term1 = np.dot(c.reshape(N, 2*H), w_1)

term2 = np.dot(q, w_2)

term3 = np.dot(result, w_3).reshape(N, M)

print (term1)
print (term2)
print (term3)

result = term1.reshape(N, 1) + term3
result = result + term2.reshape(1, M)

print (result)
