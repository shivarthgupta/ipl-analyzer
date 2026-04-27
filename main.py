import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

# =========================
# LOAD DATA (FIXED)
# =========================
path = kagglehub.dataset_download("shivarthgupta/ipl-dataset")

# Find CSV file dynamically
files = os.listdir(path)
csv_file = [f for f in files if f.endswith(".csv")][0]

df = pd.read_csv(os.path.join(path, csv_file), low_memory=False)

# Drop unwanted column safely
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print("✅ Data Loaded Successfully\n")

# =========================
# 🏏 1. TOP TEAMS
# =========================
team_runs = df.groupby('batting_team')['runs_total'].sum().sort_values(ascending=False)

print("🏏 Top Teams:\n")
print(team_runs.head(), "\n")

# =========================
# 🔥 2. TOP BATSMEN
# =========================
batsmen = df.groupby('batter')['runs_batter'].sum().sort_values(ascending=False)

print("🔥 Top Batsmen:\n")
print(batsmen.head(10), "\n")

# =========================
# 🎯 3. TOP BOWLERS
# =========================
wickets_df = df[df['wicket_kind'].notna()]
top_bowlers = wickets_df.groupby('bowler')['wicket_kind'].count().sort_values(ascending=False)

print("🎯 Top Bowlers:\n")
print(top_bowlers.head(10), "\n")

# =========================
# 🧠 4. TOSS IMPACT
# =========================
matches_df = df.drop_duplicates('match_id')

toss_result = (matches_df['toss_winner'] == matches_df['match_won_by']).value_counts()

print("🧠 Toss Impact:\n")
print(toss_result, "\n")

# =========================
# ⚡ STRIKE RATE (FIXED)
# =========================
print("\n⚡ Top Strike Rates:\n")

strike_rate = df.groupby('batter').apply(
    lambda x: (x['runs_batter'].sum() / len(x)) * 100
)

print(strike_rate.sort_values(ascending=False).head(10))

# =========================
# 💡 PLAYER IMPACT
# =========================
print("\n💡 Player Impact Score:\n")

runs = df.groupby('batter')['runs_batter'].sum()
wickets = wickets_df.groupby('bowler')['wicket_kind'].count()

impact = runs.add(wickets, fill_value=0)

print(impact.sort_values(ascending=False).head(10))

# =========================
# 🏟️ VENUES
# =========================
venues = df.groupby('venue')['runs_total'].sum().sort_values(ascending=False)

print("🏟️ Top Venues:\n")
print(venues.head(), "\n")

# =========================
# 🎯 ECONOMY (FIXED)
# =========================
print("\n🎯 Best Economy Bowlers:\n")

economy = df.groupby('bowler').apply(
    lambda x: x['runs_total'].sum() / (x['ball'].count() / 6)
)

print(economy.sort_values().head(10))

# =========================
# 📊 GRAPH (IMPROVED)
# =========================
plt.figure(figsize=(12,6))

sns.barplot(
    x=team_runs.head(5).values,
    y=team_runs.head(5).index
)

plt.title("Top 5 IPL Teams by Runs", fontsize=14)
plt.xlabel("Total Runs")
plt.ylabel("Teams")

plt.tight_layout()
plt.show()
