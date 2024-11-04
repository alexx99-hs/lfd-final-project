import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

train_data = pd.read_csv('cleaned_train.csv')
dev_data = pd.read_csv('cleaned_dev.csv')

data = pd.concat([train_data, dev_data])

X = data['cleaned_tweet']  # The cleaned tweet text
y = data['label']  # The labels ('OFF' or 'NOT')

vectorizer = CountVectorizer()  # Initialize the vectorizer
X_bow = vectorizer.fit_transform(X)  # Transform text into BoW representation

label_counts = y.value_counts()

total_samples = len(y)
prior_off = label_counts['NOT'] / total_samples  # Weight for 'OFF'
prior_not = label_counts['OFF'] / total_samples  # Weight for 'NOT'

class_prior = [prior_not, prior_off]
print("Custom Class Priors:", class_prior)

X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)

nb_model = MultinomialNB(class_prior=class_prior)
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

print("Classification Report (with Custom Class Priors):")
print(classification_report(y_test, y_pred, target_names=['NOT', 'OFF']))

print("Confusion Matrix (with Custom Class Priors):")
print(confusion_matrix(y_test, y_pred))
