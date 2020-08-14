"""
Perceptron Learning Algorithm

Apart from visualizations, this implementation should work in any dimension.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# When I hand in, then I should uncomment the below two lines:
# import sys
# data = np.genfromtxt(sys.argv[1], delimiter=',')

data = np.genfromtxt('input1.csv', delimiter=',')
m,n = data.shape

X = data[:,:n-1]
y = data[:,[n-1]]

plt.scatter(data[:,[0]][y==1], data[:,[1]][y==1])
plt.scatter(data[:,[0]][y==-1], data[:,[1]][y==-1])


ones = np.ones((m,1))
X = np.append(ones,X, axis = 1)

# Initial weights
weights = np.zeros_like(X[0])
output = weights

convergence = False
while not convergence:
    convergence = True
    for i in range(m):
        if y[i]*np.sum(weights*X[i])<=0:
            weights = weights + y[i] * X[i]
            convergence = False
    w0,w1,w2 = weights
    output = np.vstack((output, np.array([w1,w2,w0])))

np.savetxt("output1.csv", output, delimiter=",")
#np.savetxt("output1.csv", output, delimiter=",",fmt='%1.5f')


def f(x):
    w0, w1, w2 = weights
    return (-w0-x*w1)/w2

plt.plot([0,15], [f(0), f(15)])
plt.show()
