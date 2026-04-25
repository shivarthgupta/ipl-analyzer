import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/IPL.csv", low_memory=False)

# Drop unwanted column
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
wickets = df[df['wicket_kind'].notna()]
top_bowlers = wickets.groupby('bowler')['wicket_kind'].count().sort_values(ascending=False)

print("🎯 Top Bowlers:\n")
print(top_bowlers.head(10), "\n")

# =========================
# 🧠 4. TOSS IMPACT
# =========================
toss_result = (df['toss_winner'] == df['match_won_by']).value_counts()

print("🧠 Toss Impact:\n")
print(toss_result, "\n")

print("\n⚡ Top Strike Rates:\n")

strike_rate = df.groupby('batter').apply(
    lambda x: (x['runs_batter'].sum() / x['ball'].count()) * 100
)

print("\n💡 Player Impact Score:\n")

runs = df.groupby('batter')['runs_batter'].sum()
wickets = df.groupby('bowler')['wicket_kind'].count()

impact = runs.add(wickets, fill_value=0)

print(impact.sort_values(ascending=False).head(10))

print(strike_rate.sort_values(ascending=False).head(10))

# =========================
# 🏟️ 5. VENUES
# =========================
venues = df.groupby('venue')['runs_total'].sum().sort_values(ascending=False)

print("🏟️ Top Venues:\n")
print(venues.head(), "\n")
print("\n🎯 Best Economy Bowlers:\n")

economy = df.groupby('bowler').apply(
    lambda x: x['runs_total'].sum() / x['over'].nunique()
)

print(economy.sort_values().head(10))

# =========================
# 📊 6. GRAPH
# =========================
plt.figure(figsize=(10,5))
sns.barplot(x=team_runs.head(5).values, y=team_runs.head(5).index)
plt.title("Top 5 IPL Teams by Runs")
plt.xlabel("Runs")
plt.ylabel("Team")
plt.show()
