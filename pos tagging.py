import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack

nlp = spacy.load('en_core_web_sm')

train_data = pd.read_csv('cleaned_train.csv')
dev_data = pd.read_csv('cleaned_dev.csv')

data = pd.concat([train_data, dev_data])

X = data['cleaned_tweet']
y = data['label']

def get_pos_tags(text):
    doc = nlp(text)
    return ' '.join([token.pos_ for token in doc])  # Return POS tags as space-separated string

data['pos_tags'] = data['cleaned_tweet'].apply(get_pos_tags)

tfidf_word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)  # Unigrams and bigrams
X_word_ngrams = tfidf_word_vectorizer.fit_transform(X)

tfidf_char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=5000)  # Character n-grams
X_char_ngrams = tfidf_char_vectorizer.fit_transform(X)

tfidf_pos_vectorizer = TfidfVectorizer()
X_pos_tags = tfidf_pos_vectorizer.fit_transform(data['pos_tags'])

X_combined = hstack([X_word_ngrams, X_char_ngrams, X_pos_tags])

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

logreg_model = LogisticRegression(class_weight='balanced', max_iter=1000)
logreg_model.fit(X_train, y_train)

y_pred_logreg = logreg_model.predict(X_test)


print("Classification Report (Logistic Regression with POS Tags):")
print(classification_report(y_test, y_pred_logreg, target_names=['NOT', 'OFF']))

print("Confusion Matrix (Logistic Regression with POS Tags):")
print(confusion_matrix(y_test, y_pred_logreg))
