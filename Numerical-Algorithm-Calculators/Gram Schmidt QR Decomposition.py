import numpy as np

# Edit your A and b matrix here
A = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([[9], [8], [7]])

# Create matrices Q & R
m, n = np.shape(A)
Q = np.zeros((m, n))
R = np.zeros((n, n))

# Gram Schmidt QR Decomposition
for j in range(n):
    Q[:, j] = A[:, j]
    for i in range(j):
        R[i, j] = np.transpose(Q[:, j]).dot(Q[:, i])
        Q[:, j] = Q[:, j] - (R[i, j] * Q[:, i])

    R[j, j] = np.linalg.norm(Q[:, j])
    Q[:, j] = Q[:, j] / R[j, j]

print("Q = \n" + str(Q))
print("R = \n" + str(R))
