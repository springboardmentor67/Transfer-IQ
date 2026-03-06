import pandas as pd
import numpy as np
import ast
import re
from datetime import datetime

print("="*80)
print("CREATING FINAL INTEGRATED DATASET")
print("="*80)

print("\nStep 1: Loading raw files...")

injury_raw = pd.read_csv("injury_history.csv")
market_raw = pd.read_csv("market_values.csv")
tweets_raw = pd.read_csv("player_tweets.csv")
stats_raw  = pd.read_csv("statsbomb_player_stats.csv")

print(f"  ✓ Injury:  {injury_raw.shape[0]} rows")
print(f"  ✓ Market:  {market_raw.shape[0]} rows")
print(f"  ✓ Tweets:  {tweets_raw.shape[0]} rows")
print(f"  ✓ Stats:   {stats_raw.shape[0]} rows")

print("\nStep 2: Processing StatsBomb data...")

stats = stats_raw.copy()
stats.rename(columns={'player': 'player_name'}, inplace=True)

seasons = ['2019/20', '2020/21', '2021/22', '2022/23', '2023/24']
season_encoded = {s: i+1 for i, s in enumerate(seasons)}

players_base = []
for season in seasons:
    temp = stats.copy()
    temp['season'] = season
    temp['season_encoded'] = season_encoded[season]
    players_base.append(temp)

df = pd.concat(players_base, ignore_index=True)
print(f"  ✓ Created time-series: {len(df)} rows ({len(df['player_name'].unique())} players × 5 seasons)")

print("\nStep 3: Encoding positions...")

def infer_position(row):
    if row['goals'] > 10:
        return 'Forward'
    elif row['tackles_total'] > 50:
        return 'Defender'
    elif row['passes_total'] > 1500:
        return 'Midfielder'
    else:
        return 'Midfielder'

df['position'] = df.apply(infer_position, axis=1)
position_map = {'Goalkeeper': 1, 'Defender': 2, 'Midfielder': 3, 'Forward': 4}
df['position_encoded'] = df['position'].map(position_map)

df['pos_Goalkeeper'] = (df['position'] == 'Goalkeeper').astype(int)
df['pos_Defender']   = (df['position'] == 'Defender').astype(int)
df['pos_Midfielder'] = (df['position'] == 'Midfielder').astype(int)
df['pos_Forward']    = (df['position'] == 'Forward').astype(int)

print(f"  ✓ Position distribution: {df['position'].value_counts().to_dict()}")

print("\nStep 4: Adding age features...")

base_age = 25
df['current_age'] = base_age + df['season_encoded'] - 1
df['age_squared'] = df['current_age'] ** 2
df['age_decay_factor'] = np.exp(-0.05 * (df['current_age'] - 25))

def career_stage(age):
    if age < 23:
        return 'Young'
    elif age < 28:
        return 'Prime'
    elif age < 32:
        return 'Experienced'
    else:
        return 'Veteran'

df['career_stage'] = df['current_age'].apply(career_stage)

print(f"  ✓ Age range: {df['current_age'].min()}-{df['current_age'].max()}")

print("\nStep 5: Merging market values...")

def normalize_name(name):
    return str(name).strip().lower()

market = market_raw.copy()
market['player_norm'] = market['player_name'].apply(normalize_name)
df['player_norm'] = df['player_name'].apply(normalize_name)

market['date'] = pd.to_datetime(market['date'], dayfirst=True, errors='coerce')

market_latest = market.sort_values('date').groupby('player_norm').last().reset_index()
market_lookup = market_latest[['player_norm', 'market_value_eur']].set_index('player_norm')['market_value_eur'].to_dict()

df['market_value_eur'] = df['player_norm'].map(market_lookup)

def estimate_market_value(row):
    if pd.notna(row['market_value_eur']):
        return row['market_value_eur']
    performance = row['goals'] + row['assists']
    base_value = 5_000_000
    performance_bonus = performance * 1_000_000
    age_factor = 1.5 if row['current_age'] < 28 else 0.8
    return int((base_value + performance_bonus) * age_factor)

df['market_value_eur'] = df.apply(estimate_market_value, axis=1)
df['market_value_source'] = df['player_norm'].map(market_lookup).notna().map({True: 'original', False: 'estimated'})
df['log_market_value'] = np.log1p(df['market_value_eur'])

def market_tier(value):
    if value < 10_000_000:
        return 1
    elif value < 30_000_000:
        return 2
    elif value < 60_000_000:
        return 3
    else:
        return 4

df['market_value_tier_encoded'] = df['market_value_eur'].apply(market_tier)

print(f"  ✓ Matched: {(df['market_value_source'] == 'original').sum()} players with real market data")
print(f"  ✓ Estimated: {(df['market_value_source'] == 'estimated').sum()} players")

print("\nStep 6: Merging injury data...")

injury = injury_raw.copy()
injury['player_norm'] = injury['player_name'].apply(normalize_name)

injury_agg = injury.groupby('player_norm').agg(
    total_injuries=('player_name', 'count'),
    total_days_injured=('duration_days', 'sum'),
    total_matches_missed=('matches_missed', 'sum'),
    most_common_injury=('injury_type', lambda x: x.mode()[0] if len(x) > 0 else 'None')
).reset_index()

df = df.merge(injury_agg, on='player_norm', how='left')

for col in ['total_injuries', 'total_days_injured', 'total_matches_missed']:
    df[col] = df[col].fillna(0).astype(int)
df['most_common_injury'] = df['most_common_injury'].fillna('None')

df['injury_burden_index'] = df['total_injuries'] * df['total_days_injured']
df['availability_rate'] = 1 - (df['total_matches_missed'] / (df['matches'] + df['total_matches_missed']).clip(lower=1))
df['injury_frequency'] = df['total_injuries'] / df['season_encoded']

def injury_tier(count):
    if count == 0:
        return 0
    elif count <= 3:
        return 1
    elif count <= 7:
        return 2
    else:
        return 3

df['total_injuries_tier_encoded'] = df['total_injuries'].apply(injury_tier)

print(f"  ✓ Matched: {(df['total_injuries'] > 0).sum()} players with injury data")

print("\nStep 7: Merging tweet sentiment...")

tweets = tweets_raw.copy()

tweets['Sentiment'] = tweets['Sentiment'].str.lower().str.strip()

def parse_mentioned(val):
    try:
        players = ast.literal_eval(str(val))
        if isinstance(players, list):
            return [normalize_name(p) for p in players]
        else:
            return [normalize_name(str(players))]
    except:
        clean = re.sub(r"[\[\]'\"]", '', str(val)).strip()
        if clean:
            return [normalize_name(clean)]
        return []

tweets['mentioned_norm'] = tweets['mentioned_players'].apply(parse_mentioned)

sentiment_data = []
for _, row in tweets.iterrows():
    for player_norm in row['mentioned_norm']:
        sentiment_data.append({
            'player_norm': player_norm,
            'sentiment': row['Sentiment'],
            'likes': row['Number of Likes']
        })

sentiment_df = pd.DataFrame(sentiment_data)

sentiment_agg = sentiment_df.groupby('player_norm').agg(
    total_tweets=('sentiment', 'count'),
    total_likes=('likes', 'sum'),
    positive_tweets=('sentiment', lambda x: (x == 'positive').sum()),
    negative_tweets=('sentiment', lambda x: (x == 'negative').sum()),
).reset_index()

def calc_avg_sentiment(row):
    if row['total_tweets'] == 0:
        return 0
    pos = row['positive_tweets']
    neg = row['negative_tweets']
    return (pos - neg) / row['total_tweets']

sentiment_agg['avg_sentiment'] = sentiment_agg.apply(calc_avg_sentiment, axis=1)
sentiment_agg['tweet_engagement_rate'] = sentiment_agg['total_likes'] / sentiment_agg['total_tweets'].clip(lower=1)
sentiment_agg['sentiment_polarity_strength'] = abs(sentiment_agg['avg_sentiment'])
sentiment_agg['positive_tweet_ratio'] = sentiment_agg['positive_tweets'] / sentiment_agg['total_tweets'].clip(lower=1)
sentiment_agg['social_buzz_score'] = sentiment_agg['total_tweets'] * sentiment_agg['tweet_engagement_rate']

df = df.merge(sentiment_agg, on='player_norm', how='left')

for col in ['total_tweets', 'total_likes', 'positive_tweets', 'negative_tweets']:
    df[col] = df[col].fillna(0).astype(int)
for col in ['avg_sentiment', 'tweet_engagement_rate', 'sentiment_polarity_strength', 'positive_tweet_ratio', 'social_buzz_score']:
    df[col] = df[col].fillna(0).round(4)

print(f"  ✓ Matched: {(df['total_tweets'] > 0).sum()} players with tweet data")

print("\nStep 8: Engineering performance features...")

df['minutes_per_match'] = df['minutes_played'] / df['matches'].clip(lower=1)
df['goals_per90'] = (df['goals'] / df['minutes_played'].clip(lower=1)) * 90
df['assists_per90'] = (df['assists'] / df['minutes_played'].clip(lower=1)) * 90
df['shots_per90'] = (df['shots'] / df['minutes_played'].clip(lower=1)) * 90
df['goal_contributions_per90'] = df['goals_per90'] + df['assists_per90']

df['shot_conversion_rate'] = df['goals'] / df['shots'].clip(lower=1)
df['assist_to_goal_ratio'] = df['assists'] / df['goals'].clip(lower=1)
df['defensive_actions_per90'] = ((df['tackles_total'] + df['interceptions']) / df['minutes_played'].clip(lower=1)) * 90
df['dribbles_per90'] = (df['dribbles'] / df['minutes_played'].clip(lower=1)) * 90

df['attacking_output_index'] = (
    df['goals_per90'] * 3 +
    df['assists_per90'] * 2 +
    df['shots_per90'] * 0.5 +
    df['dribbles_per90'] * 0.3
)

df['tackle_success_rate'] = df['tackles_won'] / df['tackles_total'].clip(lower=1)

def tier_3(value, low, high):
    if value < low:
        return 1
    elif value < high:
        return 2
    else:
        return 3

df['minutes_played_tier_encoded'] = df['minutes_played'].apply(lambda x: tier_3(x, 1000, 2500))
df['pass_accuracy_tier_encoded'] = df['pass_accuracy_pct'].apply(lambda x: tier_3(x, 70, 85))

print(f"  ✓ Created {15} performance features")

print("\nStep 9: Creating target variable...")

df['transfer_attractiveness_score'] = (
    df['log_market_value'] * 2.0 +
    df['attacking_output_index'] * 1.5 +
    df['pass_accuracy_pct'] * 0.3 +
    (1 - df['injury_frequency']) * 10 +
    df['social_buzz_score'] * 0.05 +
    df['availability_rate'] * 5
)

print(f"  ✓ Target variable created")

print("\nStep 10: Final cleanup...")

df.drop(columns=['player_norm'], inplace=True)

float_cols = df.select_dtypes(include=['float64']).columns
for col in float_cols:
    df[col] = df[col].round(4)

col_order = [
    'player_name', 'season', 'season_encoded', 'team', 'position', 'position_encoded',
    'current_age', 'market_value_eur', 'log_market_value', 'market_value_tier_encoded',
    'market_value_source', 'age_squared', 'age_decay_factor', 'career_stage',
    'pos_Defender', 'pos_Forward', 'pos_Goalkeeper', 'pos_Midfielder',
    'matches', 'minutes_played', 'minutes_per_match', 'minutes_played_tier_encoded',
    'goals', 'assists', 'shots', 'passes_total', 'passes_complete', 'pass_accuracy_pct',
    'pass_accuracy_tier_encoded', 'tackles_total', 'tackles_won', 'tackle_success_rate',
    'dribbles', 'interceptions', 'fouls_committed',
    'goals_per90', 'assists_per90', 'shots_per90', 'goal_contributions_per90',
    'shot_conversion_rate', 'assist_to_goal_ratio', 'defensive_actions_per90',
    'dribbles_per90', 'attacking_output_index',
    'total_injuries', 'total_days_injured', 'total_matches_missed', 'most_common_injury',
    'injury_burden_index', 'availability_rate', 'injury_frequency',
    'total_injuries_tier_encoded',
    'total_tweets', 'total_likes', 'avg_sentiment', 'positive_tweets', 'negative_tweets',
    'tweet_engagement_rate', 'sentiment_polarity_strength', 'positive_tweet_ratio',
    'social_buzz_score',
    'transfer_attractiveness_score'
]

df = df[col_order]

output_path = "player_transfer_value_dataset_final.csv"
df.to_csv(output_path, index=False)

print(f"\n{'='*80}")
print(f"SUCCESS: Created {output_path}")
print(f"{'='*80}")
print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   Players: {df['player_name'].nunique()} unique")
print(f"   Seasons: {df['season'].nunique()} (2019/20 - 2023/24)")
print(f"   Features: {len(col_order)} total columns")
print(f"\n   Data sources merged:")
print(f"     • StatsBomb: {stats_raw.shape[0]} players (base)")
print(f"     • Market values: {(df['market_value_source'] == 'original').sum() // 5} players")
print(f"     • Injury data: {(df['total_injuries'] > 0).sum() // 5} players")
print(f"     • Tweet sentiment: {(df['total_tweets'] > 0).sum() // 5} players")
print(f"{'='*80}\n")