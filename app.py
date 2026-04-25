import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =========================
# 🎨 UI STYLE
# =========================
st.markdown("""
<style>
.main {background-color: #0b0f1a; color: white;}
.stMetric {background-color: #1c1f2e; padding: 15px; border-radius: 10px;}
h1, h2, h3 {color: #00ffcc;}
</style>
""", unsafe_allow_html=True)

st.title("🏏 IPL Advanced Data Analyzer")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/IPL.csv", low_memory=False)
df = df.drop(columns=['Unnamed: 0'])

matches_df = df.drop_duplicates('match_id')

teams = df['batting_team'].dropna().unique()

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Filters")

season = st.sidebar.selectbox("Season", sorted(df['season'].dropna().unique()), key="season")
venue = st.sidebar.selectbox("Venue", df['venue'].dropna().unique(), key="venue")

# =========================
# TEAM SELECTION
# =========================
team = st.selectbox("Select Team", teams, key="main_team")
team_data = df[df['batting_team'] == team]

# =========================
# TEAM STATS
# =========================
total_runs = team_data['runs_total'].sum()
matches = team_data['match_id'].nunique()
avg_score = total_runs / matches

wins_df = matches_df[matches_df['match_won_by'] == team]
wins = wins_df['match_id'].nunique()
toss_help = wins_df[wins_df['toss_winner'] == team]['match_id'].nunique()

total_matches = matches_df[
    (matches_df['batting_team'] == team) | 
    (matches_df['bowling_team'] == team)
]['match_id'].nunique()

win_pct = (wins / total_matches) * 100

# KPI CARDS
col1, col2, col3, col4 = st.columns(4)
col1.metric("🏏 Runs", total_runs)
col2.metric("🎯 Matches", matches)
col3.metric("🏆 Wins", wins)
col4.metric("📊 Win %", round(win_pct,2))

st.write(f"Toss helped win: {toss_help}")

# =========================
# PLAYER COMPARISON
# =========================
st.subheader("⚔️ Player Comparison")

players = df['batter'].dropna().unique()

p1 = st.selectbox("Player 1", players, key="p1")
p2 = st.selectbox("Player 2", players, key="p2")

def player_stats(player):
    data = df[df['batter'] == player]
    runs = data['runs_batter'].sum()
    balls = len(data)
    sr = (runs / balls) * 100 if balls > 0 else 0
    return runs, sr

r1, sr1 = player_stats(p1)
r2, sr2 = player_stats(p2)

st.write(f"{p1} → Runs: {r1}, SR: {round(sr1,2)}")
st.write(f"{p2} → Runs: {r2}, SR: {round(sr2,2)}")

# =========================
# TEAM VS TEAM
# =========================
st.subheader("🆚 Team vs Team")

team1 = st.selectbox("Team 1", teams, key="team1")
team2 = st.selectbox("Team 2", teams, key="team2")

h2h = matches_df[
    ((matches_df['batting_team'] == team1) & (matches_df['bowling_team'] == team2)) |
    ((matches_df['batting_team'] == team2) & (matches_df['bowling_team'] == team1))
]

team1_wins = h2h[h2h['match_won_by'] == team1].shape[0]
team2_wins = h2h[h2h['match_won_by'] == team2].shape[0]

st.write(f"{team1} Wins: {team1_wins}")
st.write(f"{team2} Wins: {team2_wins}")

# =========================
# TOP BATSMEN
# =========================
st.subheader("🔥 Top Batsmen")

batsmen = team_data.groupby('batter')['runs_batter'].sum().sort_values(ascending=False).head(10)
st.dataframe(batsmen)

fig = px.bar(x=batsmen.values, y=batsmen.index, orientation='h')
st.plotly_chart(fig)

# =========================
# 🏏 BOUNDARY DISTRIBUTION
# =========================
st.subheader("🏏 Boundary Distribution")

boundaries = team_data[team_data['runs_batter'].isin([4,6])]
boundary_count = boundaries['runs_batter'].value_counts()

fig = px.pie(values=boundary_count.values, names=boundary_count.index)
st.plotly_chart(fig)

# =========================
# 📈 SEASON PERFORMANCE (FINAL FIX)
# =========================
st.subheader("📈 Season Performance")

team_season = matches_df[
    (matches_df['batting_team'] == team) | 
    (matches_df['bowling_team'] == team)
].copy()

team_season = team_season.dropna(subset=['season'])
team_season['season'] = team_season['season'].astype(str)
team_season['season'] = team_season['season'].str.split('/').str[0]
team_season['season'] = pd.to_numeric(team_season['season'], errors='coerce')
team_season = team_season.dropna(subset=['season'])
team_season['season'] = team_season['season'].astype(int)

wins_by_season = team_season.groupby('season')['match_won_by'].apply(
    lambda x: (x == team).sum()
).sort_index()

fig = px.line(
    x=wins_by_season.index,
    y=wins_by_season.values,
    markers=True,
    title=f"{team} Wins by Season"
)

fig.update_layout(
    plot_bgcolor="#0b0f1a",
    paper_bgcolor="#0b0f1a",
    font=dict(color="white")
)

st.plotly_chart(fig)

# =========================
# 🤖 MATCH PREDICTION (FIXED)
# =========================
st.subheader("🤖 Match Prediction")

model_df = df[['batting_team','bowling_team','toss_winner','match_won_by']].dropna()

le_dict = {}

for col in model_df.columns:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col])
    le_dict[col] = le

X = model_df[['batting_team','bowling_team','toss_winner']]
y = model_df['match_won_by']

model = RandomForestClassifier()
model.fit(X, y)

t1 = st.selectbox("Team 1 (ML)", teams, key="ml1")
t2 = st.selectbox("Team 2 (ML)", teams, key="ml2")
toss = st.selectbox("Toss Winner", teams, key="ml3")

input_df = pd.DataFrame({
    'batting_team':[t1],
    'bowling_team':[t2],
    'toss_winner':[toss]
})

for col in input_df.columns:
    input_df[col] = le_dict[col].transform(input_df[col])

prediction = model.predict(input_df)
winner = le_dict['match_won_by'].inverse_transform(prediction)

st.success(f"Predicted Winner: {winner[0]}")