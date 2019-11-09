import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
x_train = np.random.uniform(-1,1,(10000,2))
vector = (2, -1)
y_train = []
for x,y in x_train:
    y_train.append(vector[0]*x + vector[1]*y + np.random.normal(0,1))

def cal_error(w):
    pre_on_train = [x * w[0] + y * w[1] for x, y in x_train]
    return mean_squared_error(pre_on_train, y_train)

T = 1000
w = [0, 0]
cnt = 0
plot_arr = [cal_error(w)]

def cal_matrix(x,y):
    s = 0
    for i in range(len(x)):
        s += x[i] * y[i]

    return s

for i in range(1, T):
    step_size = 1/i
    if cnt == 10:
        plot_arr.append(cal_error(w))
        cnt = 0
    n = random.randint(0,9999)
    temp = [(y_train[n] - cal_matrix(w,x_train[n]))*x_train[n][0], (y_train[n] - cal_matrix(w,x_train[n]))*x_train[n][1]]
    w = [w[0] + step_size*temp[0], w[1] + step_size*temp[1]]
    cnt += 1

plt.plot(plot_arr)
plt.xlabel("# of 10 updates")
plt.ylabel("MSE")
plt.show()