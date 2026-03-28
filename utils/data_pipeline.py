import pandas as pd
import numpy as np

def generate_mock_data(num_players=500):
    np.random.seed(42)
    
    # 1. Main player info
    player_ids = np.arange(1, num_players + 1)
    ages = np.random.randint(16, 38, size=num_players)
    contract_duration = np.random.randint(1, 6, size=num_players)
    injury_days = np.random.randint(0, 180, size=num_players)
    injury_count = np.random.randint(0, 5, size=num_players)
    sentiment_score = np.random.uniform(-1, 1, size=num_players)
    
    # Some categorical data
    positions = np.random.choice(['Forward', 'Midfielder', 'Defender', 'Goalkeeper'], size=num_players)
    
    # Nulls / duplicates injection
    # We will inject some nulls and duplicates to satisfy "Data cleaning (handle nulls, duplicates)"
    
    df_main = pd.DataFrame({
        'player_id': player_ids,
        'age': ages,
        'contract_duration': contract_duration,
        'injury_days': injury_days,
        'injury_count': injury_count,
        'sentiment_score': sentiment_score,
        'position': positions
    })
    
    # Intentionally add some duplicates
    df_main = pd.concat([df_main, df_main.head(10)]).reset_index(drop=True)
    # Intentionally add some nulls
    df_main.loc[df_main.sample(15).index, 'sentiment_score'] = np.nan
    
    # 2. Sequential Data (last 5 matches per player)
    # Even for duplicated IDs, we only need to generate for unique ones
    matches = []
    for pid in player_ids:
        pos = df_main[df_main['player_id'] == pid]['position'].iloc[0]
        
        for m in range(5):
            # Midfielders and Forwards score more, Defenders score less, GKs score 0
            if pos == 'Forward':
                goals_prob = [0.4, 0.4, 0.15, 0.05]
            elif pos == 'Midfielder':
                goals_prob = [0.7, 0.2, 0.08, 0.02]
            elif pos == 'Defender':
                goals_prob = [0.95, 0.04, 0.01, 0.0]
            else:
                goals_prob = [1.0, 0.0, 0.0, 0.0]
                
            goals = np.random.choice([0, 1, 2, 3], p=goals_prob)
            assists = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
            minutes = np.random.randint(10, 91)
            
            matches.append({
                'player_id': pid,
                'match_num': m + 1,
                'goals': goals,
                'assists': assists,
                'minutes': minutes
            })
            
    df_seq = pd.DataFrame(matches)
    
    return df_main, df_seq

def clean_and_feature_engineer(df_main, df_seq):
    # 1. Clean Main Data
    df_main = df_main.drop_duplicates(subset=['player_id']).copy()
    
    # Handle nulls
    df_main['sentiment_score'] = df_main['sentiment_score'].fillna(df_main['sentiment_score'].mean())
    df_main['injury_days'] = df_main['injury_days'].fillna(0)
    
    # 2. Group sequential data
    # "form (last 5 matches average)"
    # form could be average of goals + assists in last 5 matches
    df_seq['goal_involvement'] = df_seq['goals'] + df_seq['assists']
    form_df = df_seq.groupby('player_id')['goal_involvement'].mean().reset_index()
    form_df.rename(columns={'goal_involvement': 'form'}, inplace=True)
    
    # Calculate goals_per_match
    gpm = df_seq.groupby('player_id')['goals'].mean().reset_index()
    gpm.rename(columns={'goals': 'goals_per_match'}, inplace=True)
    
    # Calculate total stats
    totals = df_seq.groupby('player_id')[['goals', 'assists', 'minutes']].sum().reset_index()
    
    # Merge aggregations back to main
    df = pd.merge(df_main, form_df, on='player_id')
    df = pd.merge(df, gpm, on='player_id')
    df = pd.merge(df, totals, on='player_id')
    
    # Feature engineering:
    # - injury_risk = injury_days / 365
    df['injury_risk'] = df['injury_days'] / 365.0
    
    # Target generation (simulate Market Value)
    # E.g. base value + age factor + form + sentiment - injury
    # peak age around 25-28
    age_factor = 30 - abs(df['age'] - 26) 
    
    # target ranges roughly between 1m and 150m
    df['market_value'] = (
        1.0 +
        age_factor * 0.5 +
        df['form'] * 15.0 +
        (df['goals'] * 2.0) +
        df['contract_duration'] * 2.0 +
        df['sentiment_score'] * 5.0 -
        df['injury_risk'] * 20.0
    ) * 1_000_000 # scaling to millions
    
    # Add random noise
    df['market_value'] += np.random.normal(0, 2_000_000, size=len(df))
    df['market_value'] = df['market_value'].clip(lower=1_000_000) # Min 1M
    
    # Prepare Encoding + Scaling
    df = pd.get_dummies(df, columns=['position'], drop_first=True)
    
    return df, df_seq
