import pandas as pd

df = pd.read_csv('player_data.csv')
messi = df[df['player_name'].str.contains('Messi', na=False)]

if len(messi) > 0:
    latest = messi.iloc[-1]
    print('Sentiment data for Messi:')
    print(f'vader_compound_score: {latest.get("vader_compound_score", "N/A")}')
    print(f'positive_tweets: {latest.get("positive_tweets", "N/A")}')
    print(f'negative_tweets: {latest.get("negative_tweets", "N/A")}')
    print(f'total_tweets: {latest.get("total_tweets", "N/A")}')
    print(f'tweet_engagement_rate: {latest.get("tweet_engagement_rate", "N/A")}')
else:
    print('Messi not found')
    