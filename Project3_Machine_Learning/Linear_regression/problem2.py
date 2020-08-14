"""
Linear regression (no regularization)

Two featuers: Age, weight.
Last: Height
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# When I hand in, then I should uncomment the below two lines:
# import sys
# data = np.genfromtxt(sys.argv[1], delimiter=',')

data = np.genfromtxt('input2.csv', delimiter=',')
m,n = data.shape

X = data[:,:n-1]
y = data[:,[n-1]]

def plot_features():
    plt.scatter(X[:,[0]], y)
    plt.xlabel("Age")
    plt.ylabel("Height")

    plt.figure()
    plt.scatter(X[:,[1]], y)
    plt.xlabel("Weight")
    plt.ylabel("Height")
#plot_features()


#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#scaler.fit(X)
#StandardScaler(copy=True, with_mean=True, with_std=True)
#Xnorm = scaler.transform(X)
Xnorm = (X-X.mean(axis=0))/X.std(axis=0)
Xnorm = np.append(np.ones_like(y), Xnorm, axis = 1)


def hyp(x, theta):
    return np.matmul(x, theta)

def cost(X, y, theta):
    cost = np.sum((hyp(X, theta).reshape(m,1)-y)**2)/(2*m)
    return cost

def gradient(X, y, theta):
    gradient = np.zeros_like(X[0])
    for i in range(m):
        gradient += (hyp(X[i], theta) - y[i])*X[i]
    gradient /= m

    return gradient


def gradient_descent_update(X, theta, y, learning_rate):
    current_gradient = gradient(X, y, theta)
    return theta - learning_rate*current_gradient


def gradient_descent(X, theta, y, num_iterations, learning_rate):
    costs = []
    it_num = [0]
    costs.append(float(cost(X, y, theta)))

    for i in range(num_iterations):
        theta = gradient_descent_update(X, theta, y, learning_rate)
        it_num.append(i+1)
        costs.append(float(cost(X,y,theta)))
    return theta, it_num, costs

# Make required output to assignment
to_output = np.zeros([10,5])
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
for indx, alpha in enumerate(learning_rates):
      theta = np.array([0,0,0])
      theta, it_num, costs = gradient_descent(Xnorm, theta, y, 100, alpha)
      b_0, b_age, b_height = theta
      to_output[indx] = np.array([alpha, 100, b_0, b_age, b_height])

theta = np.array([0,0,0])
theta, it_num, costs = gradient_descent(Xnorm, theta, y, 1, 1)
b_0, b_age, b_height = theta
to_output[9] = np.array([1, 1, b_0, b_age, b_height])

np.savetxt("output2.csv", to_output, delimiter=",")

# PLot some of the learning rates
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
for alpha in learning_rates:
    theta = np.array([0,0,0])
    theta, it_num, costs = gradient_descent(Xnorm, theta, y, 100, alpha)
    plt.plot(it_num, costs, label = str(alpha))

plt.title("Cost reduction during gradient descent \n Comparison of different learning rates")
plt.legend()
plt.plot(it_num, costs)
plt.show()


# MY own chose of learning rate is 1, with 1 iteration!
#If I have to choose a new learning rate, it would be 0.085
