import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

train_data = pd.read_csv('cleaned_train.csv')
dev_data = pd.read_csv('cleaned_dev.csv')

data = pd.concat([train_data, dev_data])

X = data['cleaned_tweet']
y = data['label']

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],  # L1 and L2 regularization
    'solver': ['liblinear'],  # 'liblinear' solver is compatible with L1 penalty
    'class_weight': ['balanced']  # Handle class imbalance
}

logreg = LogisticRegression(max_iter=1000)

grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

best_logreg = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

y_pred_logreg = best_logreg.predict(X_test)

print("Classification Report (Fine-Tuned Logistic Regression):")
print(classification_report(y_test, y_pred_logreg, target_names=['NOT', 'OFF']))

print("Confusion Matrix (Fine-Tuned Logistic Regression):")
print(confusion_matrix(y_test, y_pred_logreg))
