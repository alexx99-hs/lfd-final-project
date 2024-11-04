import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

train_data = pd.read_csv('cleaned_train.csv')
dev_data = pd.read_csv('cleaned_dev.csv')

data = pd.concat([train_data, dev_data])

not_tweets = data[data['label'] == 'NOT']
off_tweets = data[data['label'] == 'OFF']

off_tweets_upsampled = resample(off_tweets, 
                                replace=True,  # sample with replacement
                                n_samples=len(not_tweets),  # match number of 'NOT' tweets
                                random_state=42)  # reproducible results

data_upsampled = pd.concat([not_tweets, off_tweets_upsampled])

X = data_upsampled['cleaned_tweet']
y = data_upsampled['label']

vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

print("Classification Report (with Upsampling):")
print(classification_report(y_test, y_pred, target_names=['NOT', 'OFF']))

print("Confusion Matrix (with Upsampling):")
print(confusion_matrix(y_test, y_pred))
