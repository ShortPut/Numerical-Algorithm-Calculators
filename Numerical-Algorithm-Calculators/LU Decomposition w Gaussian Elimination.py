import numpy as np

# Enter your nxn matrix here
A = np.array([[8, 5, 3], [7, -5, 3], [1, 2, -8]])
A = A.astype(float)

n = np.shape(A)[0]

L = np.identity(n)
U = A

# Perform LU decomposition using Gaussian Elimination
for k in range(n):
    for i in range(k + 1, n):
        L[i, k] = U[i, k] / U[k, k]
        for j in range(k + 1, n):
            U[i, j] = U[i, j] - (L[i, k] * U[k, j])
        U[i, k] = 0

print("U = \n" + str(U))
print("L = \n" + str(L))
