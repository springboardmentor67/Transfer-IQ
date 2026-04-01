import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("player_data.csv")

player = df[df["player_name"]=="Lionel Andrés Messi Cuccittini"]

plt.figure(figsize=(8,5))

plt.plot(player["season_encoded"], player["market_value_eur"], marker="o")

plt.title("Example Player Transfer Value Trend")

plt.xlabel("Season")
plt.ylabel("Market Value (€)")

plt.savefig("outputs/player_example_trend.png")

plt.show()