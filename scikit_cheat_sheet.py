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
#penalty = 'l2' if you want to add ridge regularization
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

##Lasso Regression
#%%
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])

##Elastic Net Regression
##Between lasso and ridge
##r = 1 is lasso, r = 0 is ridge
#%%
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])

#---------- Early Stopping ----------#
#Stop training as soon as error starts increasing again
#Error increases due to overfitting
#%%
from sklearn.base import clone
sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None,
                        learning_rate="constant", eta0=0.0005)
minimum_val_error = float("inf")
best_epoch = None
best_model = None
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
#%%
import warnings
warnings.filterwarnings('ignore')
for epoch in range(1000):
    sgd_reg.fit(X_train, y_train)
    y_val_predict = sgd_reg.predict(X_val)
    val_error = mean_squared_error(y_val_predict, y_val)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)






##Logistic Regression##
#---------- Getting the data ----------#
#%%
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())
#%%
X = iris["data"][:,3:] #Petal width
y = (iris["target"] == 2).astype(np.int) #1 if Iris-Virginica, else 0

#---------- Training the regression model ----------#
#%%
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

#---------- Predicting the petal width probabilities ----------#
#%%
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")

#%%
log_reg.predict([[1.7], [1.5]])

#In sklearn, logistic regression is regularized by l2 penalty by default
#The perameter is C not alpha. The higher C is, the lower is the regularization


#---------- Softmax Regression ----------#
#Scikit uses one-vs-all by default
#Used if logistic regression for more than 2 classes
#%%
X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
softmax_reg.fit(X, y)

softmax_reg.predict([[5, 2]])
#%%
softmax_reg.predict_proba([[5, 2]])






##Support Vector Machine##
#%%
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica
svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge")),
))
svm_clf.fit(X, y)

#%%
svm_clf.predict([[5.5, 1.7]])
#For large datasets, use SGDClassifier(loss="hinge", alpha=1/(m*C));

#----------  Non-linear SVM ----------#
#Sometimes dataset won't fit an SVM
#So we add more features to make it fit an SVM
###Always use scaling for SVM
#%%
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline((
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge"))
))
polynomial_svm_clf.fit(X, y)

#----------  Polynomial Kernel ----------#
#Above method with very high degree and still high speed
#%%
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
))
poly_kernel_svm_clf.fit(X, y)

#----------  Similarity Features with Gaussian RBF Kernel ----------#
#%%
rbf_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
))
rbf_kernel_svm_clf.fit(X, y)

#----------  SVM Regression ----------#
#%%
from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)

#----------  Non-linear SVM Regression ----------#
#%%
from sklearn.svm import SVR
svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)





##Decision Trees##
#----------  Classification ----------#
#%%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

X = iris.data[:, 2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

#Visualizing the decision tree
#%%
from sklearn.tree import export_graphviz

export_graphviz(
        tree_clf,
        out_file=image_path("iris_tree.dot"),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
)
#.dot --> .png using $ dot -Tpng iris_tree.dot -o iris_tree.png

#Predicting
#%%
tree_clf.predict_proba([[5, 1.5]]), tree_clf.predict([[5, 1.5]])

#----------  Regression ----------#
#%%
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)






##Ensemble Learning##
#----------  Voting Classifier ----------#
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC(probability=True)

voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='soft')
voting_clf.fit(X_train, y_train)

#Checking accuracy of the above ensemble
#%%
from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

#----------  Bagging and Pasting ----------#
#Baging --> with replacement
#Pasting --> without replacement
#%%
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=-1, oob_score=True)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
#For pasting, set bootstrap=False
#n_jobs is the number of CPU cores to use. -1 means all.
#oob_score tests the classifier on out-of-bag values --> bag_clf.oob_score_
#To get probabilities of each training instance, --> bag_clf.oob_decision_function_
#max_features and bootstrap_features can also be used

#----------  Random Forest Classifier ----------#
#RandromForestRegressor can also be used
#%%
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
#ExtraTreesClassifier also available

#----------  Feature Importance ----------#
#%%
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
        print(name, score)

#----------  Boosting ----------#
##AdaBoost
#%%
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5)
ada_clf.fit(X_train, y_train)

##Gradient Boosting
#%%
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y)

##Gradient boosting with ideal number of trees
#%%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred)
        for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)
#Early stopping can also be used
#GradientBoostingRegressor has a subsample hyperparameter (0 to 1).

#----------  Stacking ----------#
#https://github.com/viisar/brew





##Dimensionality Reduction##
#----------  Principal Component Analysis ----------#
#%%
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)
pca.components_
pca.explained_variance_ratio_

#To choose correct number of dimensions, set n_components to a float between 0 and 1
#0.95 would mean preserving 95% variance

#----------  Recovering original data from PCA ----------#
#%%
pca = PCA(n_components = 154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)

#----------  Incremental PCA ----------#
#%%
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
        inc_pca.partial_fit(X_batch)
        X_reduced = inc_pca.transform(X_train)

#----------  Randomized PCA ----------#
#%%
rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X_train)

#----------  Kernel PCA ----------#
#%%
from sklearn.decomposition import KernelPCA
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)

#---------- Selecting Kernel and Hyperparameters  ----------#
#%%
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
        ])
param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
        }]
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)

print(grid_search.best_params_)

#---------- Computing reconstruction pre-image error  ----------#
#%%
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433,
        fit_inverse_transform=True)
#fit_inverse_transform is responsible for this
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)

from sklearn.metrics import mean_squared_error
mean_squared_error(X, X_preimage)

#---------- LLE  ----------#
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)