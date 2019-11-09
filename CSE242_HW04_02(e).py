import numpy as np
x = [[1,3,9,2],[1,6,9,1],[1,7,7,7],[1,8,6,4],[1,1,0,8]]
t = [19,19,10,11,-3]
x = np.array(x)
temp = np.dot(x.T, x)
temp = np.dot(np.linalg.inv(temp),x.T)
res = np.dot(temp, t)
print(res)