import numpy as np

# Edit your A and b matrix here
A = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([[9], [8], [7]])

# Linear least squares by normal equations
def lls_normal(A, b):
    B = np.transpose(A).dot(A)
    y = np.transpose(A).dot(b)

    x = np.linalg.lstsq(B, y, rcond=None)
    return x


print(lls_normal(A, b))
