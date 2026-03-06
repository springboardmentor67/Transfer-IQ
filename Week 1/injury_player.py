import time
import random
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

PLAYERS = {
    8198:   "Cristiano Ronaldo",
    28003:  "Lionel Messi",
    418560: "Erling Haaland",
    342229: "Kylian Mbappe",
    581977: "Jude Bellingham",
    341048: "Vinicius Jr",
    433177: "Bukayo Saka",
    406635: "Phil Foden",
    148455: "Mohamed Salah",
    88755:  "Harry Kane",
    31909:  "Kevin De Bruyne",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

def get_injury_history(player_id, player_name):
    url = f"https://www.transfermarkt.co.in/player/verletzungen/spieler/{player_id}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        injury_table = soup.find('table', {'class': 'items'})
        
        if not injury_table:
            print(f"  ! No injury table found")
            return []
        
        records = []
        rows = injury_table.find('tbody').find_all('tr')
        
        for row in rows:
            cells = row.find_all('td')
            
            if len(cells) < 5:
                continue
            
            season = cells[0].text.strip()
            injury_type = cells[1].text.strip()
            injury_date = cells[2].text.strip()
            return_date = cells[3].text.strip()
            duration = cells[4].text.strip()
            matches_missed = cells[5].text.strip() if len(cells) > 5 else "0"
            
            duration_days = parse_duration(duration)
            
            try:
                matches_missed = int(matches_missed.split()[0]) if matches_missed else 0
            except:
                matches_missed = 0
            
            records.append({
                "player_name": player_name,
                "player_id": player_id,
                "injury_type": injury_type,
                "injury_date": injury_date,
                "return_date": return_date,
                "duration_days": duration_days,
                "matches_missed": matches_missed,
                "season": season
            })
        
        return records
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []


def parse_duration(duration_str):
    """Parse duration string to days"""
    if not duration_str or duration_str == "-":
        return None
    
    duration_str = duration_str.lower()
    
    try:
        if "day" in duration_str:
            return int(duration_str.split()[0])
        elif "month" in duration_str:
            months = int(duration_str.split()[0])
            return months * 30
        elif "year" in duration_str:
            years = int(duration_str.split()[0])
            return years * 365
    except:
        pass
    
    return None


def main():
    print("="*80)
    print("FETCHING INJURY DATA FROM TRANSFERMARKT")
    print("="*80)
    
    all_injuries = []
    
    for i, (player_id, player_name) in enumerate(PLAYERS.items(), 1):
        print(f"\n[{i}/{len(PLAYERS)}] Fetching {player_name}...")
        
        try:
            injuries = get_injury_history(player_id, player_name)
            all_injuries.extend(injuries)
            print(f"  ✓ Found {len(injuries)} injuries")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        if i < len(PLAYERS):
            sleep_time = random.uniform(2, 4)
            print(f"  Waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
    
    df = pd.DataFrame(all_injuries)
    
    if len(df) == 0:
        print("\nNo injury data found!")
        return
    
    df.to_csv("output/injury_history.csv", index=False)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n✓ Saved {len(df)} injuries to output/injury_history.csv")
    
    # Show summary
    print("\n" + "="*80)
    print("INJURY SUMMARY")
    print("="*80)
    
    print("\nInjuries per player:")
    for player, count in df['player_name'].value_counts().items():
        print(f"  {player}: {count} injuries")
    
    print("\nMost common injury types:")
    top_injuries = df['injury_type'].value_counts().head(10)
    for injury_type, count in top_injuries.items():
        print(f"  {injury_type}: {count} times")
    
    if 'duration_days' in df.columns:
        avg_duration = df['duration_days'].mean()
        print(f"\nAverage injury duration: {avg_duration:.1f} days")
    
    if 'matches_missed' in df.columns:
        total_missed = df['matches_missed'].sum()
        avg_missed = df['matches_missed'].mean()
        print(f"\nTotal matches missed: {total_missed}")
        print(f"Average matches missed per injury: {avg_missed:.1f}")
    
    print("\n" + "="*80)
    print("SAMPLE DATA (First 10 injuries)")
    print("="*80)
    print(df[['player_name', 'injury_type', 'injury_date', 'duration_days', 'matches_missed']].head(10).to_string(index=False))


if __name__ == "__main__":
    main()