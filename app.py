import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =========================
# TEAM LOGOS (LOCAL FILES)
# =========================
team_logos = {
    "Mumbai Indians": "logos/mi.png",
    "Punjab Kings": "logos/pbks.png",
    "Royal Challengers Bangalore": "logos/rcb.png",
    "Chennai Super Kings": "logos/csk.png",
    "Kolkata Knight Riders": "logos/kkr.png",
    "Delhi Capitals": "logos/dc.png",
    "Rajasthan Royals": "logos/rr.png",
    "Sunrisers Hyderabad": "logos/srh.png",
    "Gujarat Titans": "logos/gt.png",
    "Lucknow Super Giants": "logos/lsg.png"
}

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
# TEAM SELECTION (FIXED)
# =========================
team = st.selectbox("Select Team", teams, key="main_team")

# Fix name mismatch
team_fixed = team.replace("Bengaluru", "Bangalore")

team_data = df[df['batting_team'] == team]

# =========================
# LOGO DISPLAY (FIXED)
# =========================
logo_path = team_logos.get(team_fixed)

col1, col2 = st.columns([1, 4])

with col1:
    if logo_path:
        st.image(logo_path, width=100)
    else:
        st.warning("Logo not found")

with col2:
    st.title(f"{team} Dashboard")

# =========================
# TEAM STATS
# =========================
total_runs = team_data['runs_total'].sum()
matches = team_data['match_id'].nunique()

wins_df = matches_df[matches_df['match_won_by'] == team]
wins = wins_df['match_id'].nunique()

total_matches = matches_df[
    (matches_df['batting_team'] == team) | 
    (matches_df['bowling_team'] == team)
]['match_id'].nunique()

win_pct = (wins / total_matches) * 100 if total_matches > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("🏏 Runs", total_runs)
col2.metric("🎯 Matches", matches)
col3.metric("🏆 Wins", wins)
col4.metric("📊 Win %", round(win_pct, 2))

# =========================
# TEAM VS TEAM (FIXED)
# =========================
st.subheader("🆚 Team vs Team")

team1 = st.selectbox("Team 1", teams, key="team1")
team2 = st.selectbox("Team 2", teams, key="team2")

team1_fixed = team1.replace("Bengaluru", "Bangalore")
team2_fixed = team2.replace("Bengaluru", "Bangalore")

col1, col2 = st.columns(2)

with col1:
    logo1 = team_logos.get(team1_fixed)
    if logo1:
        st.image(logo1, width=100)
    st.write(team1)

with col2:
    logo2 = team_logos.get(team2_fixed)
    if logo2:
        st.image(logo2, width=100)
    st.write(team2)

# =========================
# HEAD TO HEAD
# =========================
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
# SEASON PERFORMANCE
# =========================
st.subheader("📈 Season Performance")

team_season = matches_df[
    (matches_df['batting_team'] == team) | 
    (matches_df['bowling_team'] == team)
].copy()

team_season['season'] = team_season['season'].astype(str).str.split('/').str[0]
team_season['season'] = pd.to_numeric(team_season['season'], errors='coerce')
team_season = team_season.dropna(subset=['season'])

wins_by_season = team_season.groupby('season')['match_won_by'].apply(
    lambda x: (x == team).sum()
)

fig = px.line(x=wins_by_season.index, y=wins_by_season.values, markers=True)
st.plotly_chart(fig)

# =========================
# MATCH PREDICTION
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
