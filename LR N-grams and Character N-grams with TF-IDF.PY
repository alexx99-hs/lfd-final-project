import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

train_data = pd.read_csv('cleaned_train.csv')
dev_data = pd.read_csv('cleaned_dev.csv')

data = pd.concat([train_data, dev_data])

X = data['cleaned_tweet']
y = data['label']

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),  # Include unigrams and bigrams
                                   analyzer='word',      # Word-level n-grams
                                   max_features=10000)   # Limit to top 10k features
X_word_ngrams = tfidf_vectorizer.fit_transform(X)

tfidf_char_vectorizer = TfidfVectorizer(ngram_range=(3, 5),  # Character-level n-grams (trigrams to 5-grams)
                                        analyzer='char',     # Character-level analysis
                                        max_features=5000)   # Limit to top 5k features
X_char_ngrams = tfidf_char_vectorizer.fit_transform(X)

from scipy.sparse import hstack
X_combined = hstack([X_word_ngrams, X_char_ngrams])

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

logreg_model = LogisticRegression(class_weight='balanced', max_iter=1000)
logreg_model.fit(X_train, y_train)

y_pred_logreg = logreg_model.predict(X_test)

print("Classification Report (Logistic Regression with N-grams and Character N-grams):")
print(classification_report(y_test, y_pred_logreg, target_names=['NOT', 'OFF']))

print("Confusion Matrix (Logistic Regression with N-grams and Character N-grams):")
print(confusion_matrix(y_test, y_pred_logreg))
