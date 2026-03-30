import pandas as pd
import numpy as np

# Generate synthetic dataset for player transfer value prediction
players = ["L. Messi", "C. Ronaldo", "K. Mbappe", "E. Haaland", "Vinicius Jr", "M. Salah", "K. De Bruyne", "R. Lewandowski", "H. Kane", "J. Bellingham", "P. Foden", "Pedri", "Gavi", "B. Saka", "L. Yamal"]
seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]

data = []
for p in players:
    base_value = np.random.randint(50, 150) # base market value in millions
    base_age = np.random.randint(18, 30)
    for i, s in enumerate(seasons):
        goals = np.random.randint(5, 30)
        assists = np.random.randint(2, 20)
        matches = np.random.randint(20, 50)
        age = base_age + i
        sentiment_score = np.random.uniform(0.3, 0.9)
        # Value formula: random walk with some trend based on performance
        market_value = base_value + (goals * 2) + (assists * 1) - (max(0, age - 28) * 5) + (sentiment_score * 10) + np.random.normal(0, 5)
        market_value = max(10, market_value)
        data.append({
            "player_id": players.index(p),
            "player_name": p,
            "season": s,
            "goals": goals,
            "assists": assists,
            "matches": matches,
            "age": age,
            "sentiment_score": sentiment_score,
            "market_value": market_value
        })

df = pd.DataFrame(data)
df.to_csv("e:/PROJECT/INFOSYS-AI/backend/data/dataset.csv", index=False)
print("Dataset created successfully.")
