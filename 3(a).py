from sklearn.metrics import mean_squared_error

import numpy as np
x_train = np.random.uniform(-1,1,(10000,2))
vector = (2, -1)
y_train = []
for x,y in x_train:
    y_train.append(vector[0]*x + vector[1]*y + np.random.normal(0,0.1))

x_test = np.random.uniform(-1,1,(10000,2))
vector = (2, -1)
y_test = []
for x,y in x_test:
    y_test.append(vector[0]*x + vector[1]*y + np.random.normal(0,0.1))

k = 10
x_train = x_train[:k]
y_train = y_train[:k]
coeffs = np.linalg.inv(x_train.transpose().dot(x_train)).dot(x_train.transpose()).dot(y_train)
print(coeffs)

pre_on_train = [x*coeffs[0] + y*coeffs[1] for x,y in x_train]
error_on_train = mean_squared_error(pre_on_train, y_train)
print(error_on_train)

pre_on_test = [x*coeffs[0] + y*coeffs[1] for x,y in x_test]
error_on_test = mean_squared_error(pre_on_test, y_test)
print(error_on_test)