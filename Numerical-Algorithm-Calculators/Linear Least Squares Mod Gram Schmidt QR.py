import numpy as np

# Edit your A and b matrix here
A = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([[9], [8], [7]])

# Linear least squares by Gram Schmidt QR
def lls_mod_gramschidt_qr(A, b):
    m, n = np.shape(A)
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        Q[:, j] = A[:, j]
        for i in range(j):
            R[i, j] = np.transpose(Q[:, j]).dot(Q[:, i])
            Q[:, j] = Q[:, j] - (R[i, j] * Q[:, i])

        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] = Q[:, j] / R[j, j]

    x = np.linalg.lstsq(R, (np.transpose(Q).dot(b)), rcond=None)
    return x


print(lls_mod_gramschidt_qr(A, b))
