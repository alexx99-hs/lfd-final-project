import pandas as pd
import matplotlib.pyplot as plt

column_names = ['tweet', 'label']

train_data = pd.read_csv('train.tsv', sep='\t', header=None, names=column_names)
dev_data = pd.read_csv('dev.tsv', sep='\t', header=None, names=column_names)
test_data = pd.read_csv('test.tsv', sep='\t', header=None, names=column_names)

print("Train Data Info:")
print(train_data.info())

print("\nMissing values in train data:")
print(train_data.isnull().sum())

print("\nClass Distribution in Train Data:")
print(train_data['label'].value_counts())

print("\nClass Distribution in Dev Data:")
print(dev_data['label'].value_counts())

print("\nClass Distribution in Test Data:")
print(test_data['label'].value_counts())

train_data['text_length'] = train_data['tweet'].apply(len)
dev_data['text_length'] = dev_data['tweet'].apply(len)
test_data['text_length'] = test_data['tweet'].apply(len)

print("\nTweet Length Statistics for Train Data:")
print(train_data['text_length'].describe())

plt.figure(figsize=(10, 6))
train_data['text_length'].hist(bins=50)
plt.title("Tweet Length Distribution (Train Data)")
plt.xlabel("Tweet Length")
plt.ylabel("Frequency")
plt.show()
