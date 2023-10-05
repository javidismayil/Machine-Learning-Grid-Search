import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

# =============================================================================
# 
# =============================================================================
df = pd.read_csv("D:/JAVID_ISMAYILOV_DERS/2.donem/machine-learning/vize/auto-mpg.data",
                 delim_whitespace=True,
                 header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model_year", "origin", "car_name"])
df = df.replace('?', pd.NA)
df=df.dropna()
X_=np.array(df.iloc[:, [1,2,3,4,5,6,7]])
y_=np.array(df.iloc[:,0])
X=X_.copy()
y=y_.copy()
# =============================================================================
# 
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3,
                                                    random_state=1)
sc=StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# =============================================================================
# 
# =============================================================================
svm = SVR(kernel='linear', C=0.1)
svm.fit(X_train_std, y_train)
svm.score(X_test_std, y_test)
# =============================================================================
# 
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['linear', 'rbf']}
kf = KFold(n_splits=5, shuffle=True, random_state=0)
svr = SVR()
grid_search_svr = GridSearchCV(svr, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error')
grid_search_svr.fit(X_train, y_train)
grid_search_svr.best_estimator_
grid_search_svr.best_estimator_.predict(X_test)
y_pred_svr=grid_search_svr.predict(X_test)
r2_score(y_test, y_pred_svr)
grid_search_svr.score(X_test, y_test)
# =============================================================================
# =============================================================================
# 
# =============================================================================
tree_model = DecisionTreeRegressor(random_state=42, max_depth=3, min_samples_split=5)
tree_model.fit(X_train, y_train)
tree_model.score(X_test, y_test)
# =============================================================================
# 
param_grid = {'max_depth': [3, 5, 10, 20], 'min_samples_leaf': [1, 5, 10, 20],
              'min_samples_split':[2,5,10,20]}
kf = KFold(n_splits=5, shuffle=True, random_state=0)
dt = DecisionTreeRegressor()
grid_search_tree = GridSearchCV(dt, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error')
grid_search_tree.fit(X_train, y_train)
grid_search_tree.best_estimator_
y_pred_t=grid_search_tree.predict(X_test)
r2_score(y_test, y_pred_t)
grid_search_tree.best_estimator_.score(X_test, y_test)

# =============================================================================

feature_names = ["mpg", "cylinders", "displacement", "horsepower", "weight",
       "acceleration", "model_year", "origin", "car_name"]
tree.plot_tree(tree_model,
               feature_names=feature_names,
               filled=True)
plt.show()

# =============================================================================
# 
# =============================================================================
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train_std, y_train)
knn.score(X_test_std, y_test)
# =============================================================================
# 
# =============================================================================
param_grid = {'n_neighbors': list(np.arange(1,22)), 'weights': ['uniform', 'distance']}
kf = KFold(n_splits=5, shuffle=True, random_state=4)
knn = KNeighborsRegressor()
grid_search_knn = GridSearchCV(knn, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error')
grid_search_knn.fit(X_train, y_train)
grid_search_knn.best_estimator_
grid_search_knn.score(X_test, y_test)
grid_search_knn.best_estimator_.predict(X_test)
y_pred=grid_search_knn.predict(X_test)
accuracy_score(y_test, y_pred)
r2_score(y_test, y_pred)
# =============================================================================
# 
# =============================================================================

r2_score(y_test, y_pred)
