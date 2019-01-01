
##Linear Regression##
#----------Preparing the data ----------#
#%%
import numpy as np
import matplotlib.pyplot as plt
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_new = np.array([[0], [2]])

#---------- Linear Regression ----------#
#%%
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)

#---------- Stochastic Gradient Descent ----------#
#%%
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
sgd_reg.intercept_, sgd_reg.coef_



##Polynomial Regression##
#----------Preparing the data ----------#
#%%
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

#---------- Polynomial Regression ----------#
#%%
#Convert linear feature to square
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0], X_poly[0]
#Run linear regression on the new feature
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_




#---------- Learning Curves ----------#
#%%
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")

#Learning curves for linear regression (underfitting)
#%%
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)

#Learning curves for polynomial regression (overfitting)
#%%
from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline((
                ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                ("lin_reg", LinearRegression()),
                ))
plot_learning_curves(polynomial_regression, X, y)

#---------- Regularization for overfitted data ----------#
##Ridge regression
#%%
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])