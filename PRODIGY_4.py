
import pandas as pd
import matplotlib.pyplot as plt

# Load data into a DataFrame
data = pd.read_csv('twitter_training.csv')

# Display first few rows of the dataframe
print("First few rows of the dataframe:")
print(data.head())

# Rename columns if necessary
col_names = ['ID', 'Entity', 'Sentiment', 'Content']
df = pd.read_csv('twitter_training.csv', names=col_names)
print("\nDataFrame with renamed columns:")
print(df.head())

# Data cleaning
print("\nData cleaning summary:")
print("Original shape:", df.shape)

# Check for null values
print("\nNull values per column:")
print(df.isnull().sum())

# Drop rows with null values
df.dropna(axis=0, inplace=True)
print("\nAfter dropping null values:")
print("New shape:", df.shape)

# Remove duplicates
df.drop_duplicates(inplace=True)
print("\nAfter removing duplicates:")
print("New shape:", df.shape)

# Sentiment analysis
sentiment_counts = df['Sentiment'].value_counts()
print("\nSentiment counts:")
print(sentiment_counts)

# Plot sentiment distribution
plt.figure(figsize=(6, 3))
sentiment_counts.plot(kind='bar', color=['red', 'green', 'yellow', 'blue'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=0)
plt.show()

# Brand-specific analysis (e.g., Microsoft)
brand_data = df[df['Entity'].str.contains('Microsoft', case=False)]
brand_sentiment_counts = brand_data['Sentiment'].value_counts()

# Plot brand-specific sentiment distribution
plt.figure(figsize=(6, 6))
plt.pie(brand_sentiment_counts, labels=brand_sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution for Microsoft')
plt.show()
