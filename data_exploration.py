import pandas as pd

train_data = pd.read_csv('train.tsv', sep='\t')
dev_data = pd.read_csv('dev.tsv', sep='\t')
test_data = pd.read_csv('test.tsv', sep='\t')

print("Train Data:")
print(train_data.head())

print("\nDev Data:")
print(dev_data.head())

print("\nTest Data:")
print(test_data.head())
