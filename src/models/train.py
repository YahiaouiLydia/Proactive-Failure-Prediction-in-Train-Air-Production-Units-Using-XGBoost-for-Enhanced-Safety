# src/models/train.py

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import GridSearchCV

def tune_xgboost_model(X_train, y_train, param_grid):
    model = xgb.XGBClassifier(objective='binary:logistic', seed=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"Best XGBoost parameters: {grid_search.best_params_}")
    print(f"Best XGBoost score: {grid_search.best_score_}")
    return grid_search.best_estimator_

def build_xgboost_model(X_train, y_train):
    model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=50, seed=42)
    model.fit(X_train, y_train)
    return model

def build_rf_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    return model

def build_dt_model(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=3, random_state=42)  
    model.fit(X_train, y_train)
    return model

def build_svm_model(X_train, y_train):
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model

def build_knn_model(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5)  # Default n_neighbors=5
    model.fit(X_train, y_train)
    return model

def build_logistic_model(X_train, y_train):
    model = LogisticRegression(random_state=42)  # max_iter=1000 pour convergence
    model.fit(X_train, y_train)
    return model

def build_nb_model(X_train, y_train): 
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


