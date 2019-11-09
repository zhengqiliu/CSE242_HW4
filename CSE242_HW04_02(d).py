import random
x_train = [[1,3,9,2],[1,6,9,1]]
y_train = [19,19]
w = [1,1,1,1]

def cal_matrix(x,y):
    s = 0
    for i in range(len(x)):
        s += x[i] * y[i]

    return s

step_size = 0.01
for i in range(2):
    print(w)
    temp = []
    for m in range(4):
        temp.append((y_train[i] - cal_matrix(x_train[i],w))*x_train[i][m])
    w = [w[0] + step_size*temp[0], w[1] + step_size*temp[1], w[2] + step_size*temp[2], w[3] + step_size*temp[3]]

print(w)