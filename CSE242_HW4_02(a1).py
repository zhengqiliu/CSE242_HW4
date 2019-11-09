from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import math
x_train = [[3,9,2],[6,9,1],[7,7,7],[8,6,4],[1,0,8]]
y_train = [19,19,10,11,-3]

model = linear_model.LinearRegression(fit_intercept=False)
model.fit(x_train, y_train)
print(model.coef_)

pre = model.predict(x_train)
error = mean_squared_error(pre, y_train)
print(math.sqrt(error))