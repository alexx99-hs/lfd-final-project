import pandas as pd
import re

column_names = ['tweet', 'label']
train_data = pd.read_csv('train.tsv', sep='\t', header=None, names=column_names)
dev_data = pd.read_csv('dev.tsv', sep='\t', header=None, names=column_names)
test_data = pd.read_csv('test.tsv', sep='\t', header=None, names=column_names)

def clean_tweet(tweet):
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'http\S+|www\S+', '', tweet)
    tweet = tweet.lower()
    tweet = tweet.strip()
    return tweet

train_data['cleaned_tweet'] = train_data['tweet'].apply(clean_tweet)
dev_data['cleaned_tweet'] = dev_data['tweet'].apply(clean_tweet)
test_data['cleaned_tweet'] = test_data['tweet'].apply(clean_tweet)

print("Original Tweet:", train_data['tweet'].iloc[0])
print("Cleaned Tweet:", train_data['cleaned_tweet'].iloc[0])

train_data.to_csv('cleaned_train.csv', index=False)
dev_data.to_csv('cleaned_dev.csv', index=False)
test_data.to_csv('cleaned_test.csv', index=False)

