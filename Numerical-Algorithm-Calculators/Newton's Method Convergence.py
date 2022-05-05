import numpy as np

# Enter your functions here
def f(x):
    return 2 * np.cosh(x / 4) - x


def fprime(x):
    return 0.5 * np.sinh(x / 4) - 1


# Newton's Method setup
x0 = 3
max_iterations = 50

itn = 0
not_done = True
xk_plus1 = x0
x = [xk_plus1]
tolerance = 10 ** (-8)

# Newton's Method
while (not_done) and (itn < max_iterations):
    itn = itn + 1
    xk = xk_plus1
    xk_plus1 = xk - (f(xk) / fprime(xk))
    if (np.absolute(xk_plus1 - xk) < tolerance) or xk_plus1 == np.Infinity:
        not_done = False
    x.append(xk_plus1)

print("iterations: " + str(itn))
print("x = " + str(x))

# Compute absolute errors for each iteration assuming last itn is solution
ek_newton = []
for i in x:
    ek_newton.append(np.absolute(i - x[-1]))

print("absolute error = " + str(ek_newton))
