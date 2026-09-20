import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Metis · Player Scouting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG_COLOR     = "#0a0e1a"   # deep navy
CARD_COLOR   = "#111827"   # slightly lighter panel
BORDER_COLOR = "#1e2d45"   # subtle blue-steel border
GOLD         = "#c9a84c"   # Metis gold — primary accent
GOLD_DIM     = "#7a5f28"   # dimmed gold for grid lines
TEXT_COLOR   = "#d4dbe8"   # cool off-white
TEXT_DIM     = "#6b7a96"   # muted secondary text
TARGET_COLOR = "#e05c5c"   # target player — warm red
COMP_COLOR   = "#4db8d4"   # comparison player — teal
ZERO_COLOR   = "#2e3f58"   # zero-line on bar chart

# ── Global CSS injection ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Base ── */
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {BG_COLOR};
      color: {TEXT_COLOR};
  }}
  [data-testid="stSidebar"] {{
      background-color: {CARD_COLOR};
      border-right: 1px solid {BORDER_COLOR};
  }}
  [data-testid="stSidebar"] * {{ color: {TEXT_COLOR} !important; }}

  /* ── Wordmark ── */
  .metis-wordmark {{
      font-family: 'Georgia', serif;
      font-size: 2.6rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      color: {GOLD};
      margin: 0;
      line-height: 1;
  }}
  .metis-sub {{
      font-family: 'Georgia', serif;
      font-size: 0.78rem;
      letter-spacing: 0.28em;
      text-transform: uppercase;
      color: {TEXT_DIM};
      margin-top: 0.2rem;
  }}
  .metis-rule {{
      border: none;
      border-top: 1px solid {GOLD_DIM};
      margin: 0.6rem 0 1.4rem 0;
  }}

  /* ── Sidebar label ── */
  .sidebar-brand {{
      font-family: 'Georgia', serif;
      font-size: 1.05rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: {GOLD};
      margin-bottom: 0.8rem;
  }}

  /* ── Section headers ── */
  .section-label {{
      font-size: 0.68rem;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: {TEXT_DIM};
      margin-bottom: 0.3rem;
  }}

  /* ── Player card ── */
  .player-card {{
      background: {CARD_COLOR};
      border: 1px solid {BORDER_COLOR};
      border-left: 3px solid {GOLD};
      border-radius: 6px;
      padding: 0.9rem 1.2rem;
      margin-bottom: 1rem;
  }}
  .player-name {{
      font-family: 'Georgia', serif;
      font-size: 1.35rem;
      font-weight: 600;
      color: {TEXT_COLOR};
      letter-spacing: 0.04em;
  }}
  .player-meta {{
      font-size: 0.8rem;
      color: {TEXT_DIM};
      margin-top: 0.25rem;
      letter-spacing: 0.04em;
  }}
  .player-meta span {{
      margin-right: 1.2rem;
  }}

  /* ── Metric pill row ── */
  .pill-row {{
      display: flex;
      gap: 0.55rem;
      flex-wrap: wrap;
      margin-top: 0.6rem;
  }}
  .pill {{
      background: {BG_COLOR};
      border: 1px solid {BORDER_COLOR};
      border-radius: 20px;
      padding: 0.2rem 0.75rem;
      font-size: 0.72rem;
      color: {TEXT_DIM};
      white-space: nowrap;
  }}
  .pill b {{ color: {TEXT_COLOR}; font-weight: 600; }}

  /* ── Similarity badge ── */
  .sim-badge {{
      display: inline-block;
      background: {GOLD};
      color: #0a0e1a;
      font-weight: 700;
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      padding: 0.15rem 0.6rem;
      border-radius: 12px;
      margin-left: 0.5rem;
      vertical-align: middle;
  }}

  /* ── Match table ── */
  .match-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.55rem 0.9rem;
      border-bottom: 1px solid {BORDER_COLOR};
      font-size: 0.84rem;
  }}
  .match-row:last-child {{ border-bottom: none; }}
  .match-row:hover {{ background: {BORDER_COLOR}22; }}
  .match-name {{ color: {TEXT_COLOR}; font-weight: 500; }}
  .match-detail {{ color: {TEXT_DIM}; font-size: 0.75rem; }}
  .sim-bar-wrap {{
      width: 80px;
      background: {BG_COLOR};
      border-radius: 4px;
      height: 5px;
      overflow: hidden;
  }}
  .sim-bar {{
      height: 5px;
      background: {GOLD};
      border-radius: 4px;
  }}

  /* ── Streamlit overrides ── */
  [data-testid="stDataFrame"] {{ border: 1px solid {BORDER_COLOR} !important; }}
  .stSelectbox label, .stMultiSelect label, .stSlider label {{
      font-size: 0.72rem !important;
      letter-spacing: 0.1em !important;
      text-transform: uppercase !important;
      color: {TEXT_DIM} !important;
  }}
  div[data-baseweb="tab-list"] {{ border-bottom: 1px solid {BORDER_COLOR}; }}
  button[data-baseweb="tab"] {{
      color: {TEXT_DIM} !important;
      font-size: 0.78rem !important;
      letter-spacing: 0.08em !important;
  }}
  button[aria-selected="true"] {{
      color: {GOLD} !important;
      border-bottom: 2px solid {GOLD} !important;
  }}
  [data-testid="stMetric"] {{
      background: {CARD_COLOR};
      border: 1px solid {BORDER_COLOR};
      border-radius: 6px;
      padding: 0.5rem 0.75rem;
  }}
  [data-testid="stMetricLabel"] {{ color: {TEXT_DIM} !important; font-size: 0.7rem !important; }}
  [data-testid="stMetricValue"] {{ color: {TEXT_COLOR} !important; }}
  hr {{ border-color: {BORDER_COLOR} !important; }}
</style>
""", unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_parquet("scouting_features.parquet")
    df = df.dropna(subset=["player", "league", "season"]).copy()
    df["player"] = df["player"].astype(str).str.strip()
    df["team"]   = df["team"].astype(str).str.strip()
    df["league"] = df["league"].astype(str).str.strip()
    season_str   = df["season"].astype(str).str.replace(".0", "", regex=False).str.strip()
    df["season"] = season_str.replace({
        "2425": "24-25", "2526": "25-26",
        "2024-2025": "24-25", "2025-2026": "25-26",
    })
    return df

df           = load_data()
scaled_cols  = [c for c in df.columns if c.startswith("scaled_")]
raw_feature_cols = [c.replace("scaled_", "") for c in scaled_cols]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚡ Metis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Scouting Filters</div>', unsafe_allow_html=True)

    seasons_available = sorted(df["season"].unique())
    selected_seasons  = st.multiselect("Season", options=seasons_available, default=seasons_available)

    leagues_available = sorted(df["league"].unique())
    selected_leagues  = st.multiselect("League", options=leagues_available, default=leagues_available)

    min_minutes = st.slider("Min. Minutes Played", 600, int(df["Playing Time_Min"].max()), 900, 100)
    top_n       = st.slider("Matches to Return",   3,   15,                                5)

    st.markdown("---")
    st.markdown(f'<div style="font-size:0.68rem;color:{TEXT_DIM};letter-spacing:0.06em">'
                f'Europe · Top 5 Leagues · Per-90 Cosine Similarity</div>', unsafe_allow_html=True)

filtered_df = df[
    df["season"].isin(selected_seasons) &
    df["league"].isin(selected_leagues) &
    (df["Playing Time_Min"] >= min_minutes)
].copy()


# ── Wordmark header ───────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 0.5rem 0 0 0;">
  <div class="metis-wordmark">METIS</div>
  <div class="metis-sub">Football Intelligence · Player Similarity Engine</div>
  <hr class="metis-rule">
</div>
""", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("No players match the current filters.")
    st.stop()


# ── Player selector ───────────────────────────────────────────────────────────

st.markdown('<div class="section-label">Select a Player to Scout</div>', unsafe_allow_html=True)
selected_player_name = st.selectbox(
    "Player",
    options=sorted(filtered_df["player"].unique()),
    index=None,
    placeholder="Search by name…",
    label_visibility="collapsed",
)

if not selected_player_name:
    st.markdown(
        f'<div style="color:{TEXT_DIM};font-size:0.9rem;padding:1rem 0">'
        f'Search for a player above to generate their statistical profile and similarity matches.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

target_player = (
    filtered_df[filtered_df["player"] == selected_player_name]
    .sort_values("season", ascending=False)
    .iloc[0]
)


# ── Target player card ────────────────────────────────────────────────────────

league_short = target_player["league"].replace("ENG-", "").replace("ESP-", "").replace(
    "FRA-", "").replace("ITA-", "").replace("GER-", "")

st.markdown(f"""
<div class="player-card">
  <div class="player-name">{target_player['player']}</div>
  <div class="player-meta">
    <span>{target_player['team']}</span>
    <span>{league_short}</span>
    <span>{target_player['pos']}</span>
    <span>{target_player['season']}</span>
    <span>{int(target_player['Playing Time_Min'])} min</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# ── Similarity model ──────────────────────────────────────────────────────────

candidate_pool = filtered_df[filtered_df["primary_pos"] == target_player["primary_pos"]].copy()
if len(candidate_pool) <= 1:
    st.info("Not enough players in this positional group to generate matches.")
    st.stop()

target_vector = target_player[scaled_cols].values.reshape(1, -1)
candidate_pool["similarity"] = cosine_similarity(target_vector, candidate_pool[scaled_cols].values)[0]

top_matches = (
    candidate_pool[candidate_pool["player"].str.lower() != selected_player_name.lower()]
    .sort_values("similarity", ascending=False)
    .head(top_n)
)
top_match = top_matches.iloc[0]


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _feature_label(col: str) -> str:
    return {
        "goals_np_p90":           "Goals (np)",
        "xg_p90":                 "xG",
        "shots_p90":              "Shots",
        "shots_on_target_p90":    "SoT",
        "shot_accuracy":          "Shot Acc.",
        "assists_p90":            "Assists",
        "xag_p90":                "xAG",
        "key_passes_p90":         "Key Passes",
        "passes_final_third_p90": "Final 3rd Pass",
        "progressive_passes_p90": "Prog. Passes",
        "progressive_carries_p90":"Prog. Carries",
        "pass_completion_pct":    "Pass Cmp%",
        "dribbles_p90":           "Dribbles",
        "dribble_success_pct":    "Dribble Suc%",
        "tackles_won_p90":        "Tackles Won",
        "interceptions_p90":      "Interceptions",
        "pressures_p90":          "Pressures",
        "blocks_p90":             "Blocks",
        "crosses_p90":            "Crosses",
        "fouls_drawn_p90":        "Fouls Drawn",
    }.get(col, col.replace("_p90", "").replace("_", " ").title())

LABELS = [_feature_label(f) for f in raw_feature_cols]


def make_radar(target, match, pool) -> plt.Figure:
    n      = len(raw_feature_cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist() + [0]
    mn, mx = pool[raw_feature_cols].min(), pool[raw_feature_cols].max()

    def _norm(row):
        v = ((row[raw_feature_cols] - mn) / (mx - mn + 1e-6)).clip(0, 1).tolist()
        return v + v[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    for vals, color, label in [
        (_norm(target), TARGET_COLOR, f"{target['player']} ({target['season']})"),
        (_norm(match),  COMP_COLOR,   f"{match['player']} ({match['season']})"),
    ]:
        ax.plot(angles, vals, color=color, linewidth=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(LABELS, size=8, color=TEXT_COLOR)
    ax.set_yticklabels([])
    ax.spines["polar"].set_color(GOLD_DIM)
    ax.grid(color=GOLD_DIM, linestyle="--", alpha=0.4)
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.3, 1.25),
        facecolor=CARD_COLOR, edgecolor=BORDER_COLOR,
        labelcolor=TEXT_COLOR, fontsize=8,
    )
    fig.tight_layout()
    return fig


def make_delta_bar(target, match) -> plt.Figure:
    deltas = (
        target[scaled_cols].values.astype(float)
        - match[scaled_cols].values.astype(float)
    )
    order  = np.argsort(np.abs(deltas))
    labels = [LABELS[i] for i in order]
    values = deltas[order]
    colors = [TARGET_COLOR if v >= 0 else COMP_COLOR for v in values]

    n   = len(labels)
    h   = max(3.5, n * 0.38)
    fig, ax = plt.subplots(figsize=(5.5, h))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.barh(range(n), values, color=colors, height=0.62, zorder=3)
    ax.axvline(0, color=ZERO_COLOR, linewidth=1.2, zorder=2)

    for i, v in enumerate(values):
        ax.text(
            v + (0.06 if v >= 0 else -0.06), i, f"{v:+.2f}",
            va="center", ha=("left" if v >= 0 else "right"),
            color=TEXT_COLOR, fontsize=8, zorder=4,
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, color=TEXT_COLOR, fontsize=9)
    ax.tick_params(axis="x", colors=TEXT_DIM, labelsize=8)
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", color=BORDER_COLOR, linestyle="--", alpha=0.6, zorder=1)

    lim = max(np.abs(values).max() * 1.35, 0.5)
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f"))

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=TARGET_COLOR, label=f"{target['player']} higher"),
            Patch(facecolor=COMP_COLOR,   label=f"{match['player']} higher"),
        ],
        loc="lower right",
        facecolor=CARD_COLOR, edgecolor=BORDER_COLOR,
        labelcolor=TEXT_COLOR, fontsize=8,
    )
    ax.set_xlabel("z-score difference  (target − match)", color=TEXT_DIM, fontsize=8)
    fig.tight_layout()
    return fig


# ── Layout: results + charts ──────────────────────────────────────────────────

col_results, col_charts = st.columns([1, 1], gap="large")

with col_results:
    st.markdown(
        f'<div class="section-label">Top {top_n} Statistical Twins · {target_player["primary_pos"]}</div>',
        unsafe_allow_html=True,
    )

    # Custom match list
    rows_html = ""
    for _, row in top_matches.iterrows():
        sim      = row["similarity"]
        sim_pct  = f"{sim * 100:.1f}%"
        bar_w    = int(sim * 100)
        p_league = row["league"].replace("ENG-","").replace("ESP-","").replace(
                   "FRA-","").replace("ITA-","").replace("GER-","")
        rows_html += f"""
        <div class="match-row">
          <div>
            <div class="match-name">{row['player']}</div>
            <div class="match-detail">{row['team']} · {p_league} · {row['season']}</div>
          </div>
          <div style="text-align:right">
            <div style="font-size:0.8rem;color:{TEXT_COLOR};font-weight:600">{sim_pct}</div>
            <div class="sim-bar-wrap"><div class="sim-bar" style="width:{bar_w}%"></div></div>
          </div>
        </div>"""

    st.markdown(
        f'<div style="background:{CARD_COLOR};border:1px solid {BORDER_COLOR};'
        f'border-radius:6px;overflow:hidden">{rows_html}</div>',
        unsafe_allow_html=True,
    )

    # Per-90 comparison table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-label">Per-90 Stats · '
        f'{target_player["player"]} vs {top_match["player"]}</div>',
        unsafe_allow_html=True,
    )
    comp_df = pd.DataFrame({
        "Metric":                [_feature_label(f) for f in raw_feature_cols],
        target_player["player"]: [round(target_player[f], 3) for f in raw_feature_cols],
        top_match["player"]:     [round(top_match[f], 3)     for f in raw_feature_cols],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)


with col_charts:
    match_options = {
        f"{row['player']}  ({row['team']}, {row['season']})": idx
        for idx, row in top_matches.iterrows()
    }
    selected_match_label = st.selectbox(
        "Compare against",
        options=list(match_options.keys()),
        label_visibility="collapsed",
    )
    selected_match = top_matches.loc[match_options[selected_match_label]]

    st.markdown(
        f'<div style="font-size:0.75rem;color:{TEXT_DIM};margin-bottom:0.5rem">'
        f'Comparing <b style="color:{TEXT_COLOR}">{target_player["player"]}</b> against '
        f'<b style="color:{TEXT_COLOR}">{selected_match["player"]}</b>'
        f'<span class="sim-badge">{selected_match["similarity"]*100:.1f}% match</span></div>',
        unsafe_allow_html=True,
    )

    tab_radar, tab_delta = st.tabs(["Radar Profile", "Stat Difference"])

    with tab_radar:
        st.pyplot(make_radar(target_player, selected_match, candidate_pool))
        st.caption("Values normalised 0–1 within position group. Shape = style, size = output.")

    with tab_delta:
        st.pyplot(make_delta_bar(target_player, selected_match))
        st.caption("z-score difference (target − match). Sorted by magnitude — biggest gaps at top.")

