from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
x_train = [[1,3,9,2],[1,6,9,1],[1,7,7,7],[1,8,6,4],[1,1,0,8]]
y_train = [19,19,10,11,-3]

model = Ridge(fit_intercept=False, alpha=0.3)
model.fit(x_train,y_train)
print(model.coef_)
pre = model.predict(x_train)
error = mean_squared_error(pre, y_train)
print(error)