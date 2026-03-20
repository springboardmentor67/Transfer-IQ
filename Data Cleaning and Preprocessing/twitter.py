
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon (run once)
nltk.download('vader_lexicon')

# ===============================
# 1. LOAD DATASET
# ===============================

file_path = "Football player analyzer AI\\Twitter dataset\\2020-07-09 till 2020-09-19.csv"  # change if needed
df = pd.read_csv(file_path)

print("Dataset Loaded Successfully")
print(df.head())
print("\nColumns in dataset:", df.columns)

# ===============================
# 2. TEXT CLEANING FUNCTION
# ===============================

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)     # remove mentions
    text = re.sub(r"#", "", text)        # remove hashtag symbol
    text = re.sub(r"[^A-Za-z\s]", "", text)  # remove special characters
    text = text.lower()
    return text

# Change 'text' to your actual tweet column name if different
df['clean_text'] = df['text'].apply(clean_text)

# ===============================
# 3. APPLY VADER SENTIMENT
# ===============================

sia = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    return sia.polarity_scores(text)

df['sentiment_scores'] = df['clean_text'].apply(get_sentiment_scores)

# Extract compound score
df['compound_score'] = df['sentiment_scores'].apply(lambda x: x['compound'])

# ===============================
# 4. CLASSIFY SENTIMENT
# ===============================

def classify_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['compound_score'].apply(classify_sentiment)

print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())

# ===============================
# 5. VISUALIZATION
# ===============================

plt.figure()
sns.countplot(x='sentiment', data=df)
plt.title("Sentiment Distribution using VADER")
plt.show()

# ===============================
# 6. SAVE PROCESSED DATA
# ===============================

df.to_csv("sentiment.csv", index=False)

print("\nProcessed file saved as 'tweets_with_vader_sentiment.csv'")
