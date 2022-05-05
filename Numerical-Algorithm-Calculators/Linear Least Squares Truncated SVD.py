import numpy as np

# Edit your A and b matrix here
A = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([[9], [8], [7]])

condition = 10 ** 8

# Linear least squares by Truncated SVD
def lls_tsvd(A, b):
    U, S, V = np.linalg.svd(A)
    S = np.diag(S)

    kappa = np.transpose(S[0, 0] / np.diag(S))
    r = np.sum(kappa <= condition)

    y = np.linalg.lstsq(S[0:r, 0:r], np.transpose(U[:, 0:r]).dot(b), rcond=None)
    x = V[:, 0:r].dot(y[0])
    return x


print(lls_tsvd(A, b))
