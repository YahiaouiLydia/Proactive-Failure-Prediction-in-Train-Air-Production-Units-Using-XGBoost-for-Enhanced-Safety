# src/models/evaluate.py

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def compute_cv_scores(model, X_train, y_train, cv_folds=5):
    return cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')

def make_predictions(model, X_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_proba

def calculate_metrics(y_test, y_pred, y_pred_proba):
    return {
        'test_accuracy': accuracy_score(y_test, y_pred) * 100,
        'precision': precision_score(y_test, y_pred, average='weighted') * 100,
        'recall': recall_score(y_test, y_pred, average='weighted') * 100,
        'f1': f1_score(y_test, y_pred, average='weighted') * 100,
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

def generate_classification_report(y_test, y_pred):
    return classification_report(y_test, y_pred, output_dict=True)

def evaluate_model(model, X_train, y_train, X_test, y_test, cv_folds=5):
    cv_scores = compute_cv_scores(model, X_train, y_train, cv_folds)
    y_pred, y_pred_proba = make_predictions(model, X_test)
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    metrics['cv_accuracy_mean'] = cv_scores.mean() * 100
    metrics['cv_accuracy_std'] = cv_scores.std() * 100
    report = generate_classification_report(y_test, y_pred)
    return metrics, report, y_pred, y_pred_proba