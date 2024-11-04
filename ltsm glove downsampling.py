import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample  # For downsampling
import nltk
nltk.download('punkt')

train_data = pd.read_csv('cleaned_train.csv')
dev_data = pd.read_csv('cleaned_dev.csv')

data = pd.concat([train_data, dev_data])

X = data['cleaned_tweet']
y = data['label']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

sequences = tokenizer.texts_to_sequences(X)
max_length = max([len(seq) for seq in sequences])
X_padded = pad_sequences(sequences, maxlen=max_length, padding='post')

embedding_index = {}
with open('glove.twitter.27B.100d.txt', 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

embedding_dim = 100  # GloVe Twitter 100d
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

y_train = np.where(y_train == 'OFF', 1, 0)
y_test = np.where(y_test == 'OFF', 1, 0)

train_data = pd.DataFrame({'tweet': list(X_train), 'label': y_train})

train_majority = train_data[train_data.label == 0]  # NOT
train_minority = train_data[train_data.label == 1]  # OFF

train_majority_downsampled = resample(train_majority, 
                                      replace=False,  # Sample without replacement
                                      n_samples=len(train_minority),  # Match the number of minority samples
                                      random_state=42)  # Reproducibility

train_downsampled = pd.concat([train_majority_downsampled, train_minority])

X_train_downsampled = np.stack(train_downsampled['tweet'])
y_train_downsampled = train_downsampled['label'].values

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_downsampled, y_train_downsampled, epochs=10, batch_size=64, validation_split=0.2)

y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

print("Classification Report (LSTM with Downsampled Data and GloVe Twitter Embeddings):")
print(classification_report(y_test, y_pred, target_names=['NOT', 'OFF']))
