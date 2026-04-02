import pandas as pd

df = pd.read_csv('player_data.csv')
messi = df[df['player_name'].str.contains('Messi', na=False)]

print('=' * 50)
print('MESSI DATA CHECK')
print('=' * 50)

if len(messi) > 0:
    print(f"\nFound {len(messi)} records for Messi")
    latest = messi.iloc[-1]
    
    print("\nLatest season data:")
    print(f"  season: {latest['season']}")
    print(f"  vader_compound_score: {latest['vader_compound_score']}")
    print(f"  positive_count: {latest['positive_count']}")
    print(f"  negative_count: {latest['negative_count']}")
    print(f"  neutral_count: {latest['neutral_count']}")
    print(f"  sentiment_label: {latest['sentiment_label']}")
    
    print("\nAll seasons for Messi:")
    for idx, row in messi.iterrows():
        print(f"  {row['season']}: vader_compound_score = {row['vader_compound_score']}")
else:
    print("Messi not found")

print("\n" + "=" * 50)