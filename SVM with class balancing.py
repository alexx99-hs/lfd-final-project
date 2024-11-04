import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

train_data = pd.read_csv('cleaned_train.csv')
dev_data = pd.read_csv('cleaned_dev.csv')

data = pd.concat([train_data, dev_data])

X = data['cleaned_tweet']
y = data['label']

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

svm_model = SVC(class_weight='balanced')  # Automatically adjust class weights
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print("Classification Report (SVM with Class Weighting):")
print(classification_report(y_test, y_pred, target_names=['NOT', 'OFF']))

print("Confusion Matrix (SVM with Class Weighting):")
print(confusion_matrix(y_test, y_pred))
