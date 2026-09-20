# ⚡ Metis — Football Intelligence · Player Similarity Engine

Metis finds players across Europe's top 5 leagues who play the most similarly to any target player, based on their per-90 statistical profile. Search for Bukayo Saka, get back the ten players whose on-pitch output most closely mirrors his — across all positions, leagues, and seasons.

---

## How it works

1. **Data acquisition** — `data_pipeline.py` fetches two seasons of player stats (2024-25 and 2025-26) from FBref via soccerdata (standard, shooting, misc tables) and directly scrapes the passing, possession, and defense tables from FBref's Big 5 combined pages using the same Selenium session to bypass Cloudflare.

2. **Feature engineering** — `prepare_features.py` collapses mid-season transfers into single rows, converts all counting stats to per-90 rates, and computes derived metrics (shot accuracy, pass completion %, dribble success %). Players with fewer than 600 minutes and goalkeepers are excluded.

3. **Similarity model** — Features are standardised with `StandardScaler` fitted separately per positional group (FW / MF / DF), so a forward's output is compared against other forwards' distributions, not the full population. Similarity is computed with cosine distance on the scaled feature vectors, which captures the *shape* of a player's output profile rather than raw volume.

4. **Dashboard** — A Streamlit app lets you search for any player, see their top N statistical twins ranked by similarity score, and compare profiles with a radar chart and a z-score divergence bar chart.

---

## Feature set

All features are per 90 minutes unless noted.

| Category | Features |
|---|---|
| **Attack** | Non-penalty goals, shots, shots on target, shot accuracy (ratio) |
| **Creation** | Assists, xG, xAG, key passes, passes into the final third |
| **Progression** | Progressive passes, progressive carries, pass completion % |
| **Dribbling** | Dribbles completed, dribble success % |
| **Defending** | Tackles won, interceptions, pressures, blocks |
| **General** | Crosses, fouls drawn |

Core features (goals, assists, shots, tackles, interceptions, crosses, fouls drawn, shot accuracy) are always available from the standard/shooting/misc tables. Enriched features (xG, xAG, progressive passes/carries, key passes, dribbles, pressures, blocks) are pulled from the passing, possession, and defense tables and activate automatically once the full pipeline has run. `prepare_features.py` detects which features have populated data and silently excludes any that are still zero.

---

## Data

| | |
|---|---|
| **Source** | FBref (via soccerdata + direct scrape) |
| **Leagues** | Premier League, La Liga, Serie A, Bundesliga, Ligue 1 |
| **Seasons** | 2024-25, 2025-26 |
| **Players** | ~3,400 after 600-minute filter and GK exclusion |
| **Positions** | FW (534), MF (1,663), DF (1,218) |

---

## Project structure

```
metis/
├── app.py                  # Streamlit dashboard
├── data_pipeline.py        # FBref acquisition — writes top5_raw_2024_2026.csv
├── prepare_features.py     # Feature engineering — writes scouting_features.parquet
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup

**Requirements**

- Python 3.11+
- Google Chrome (used by soccerdata's Selenium session to handle Cloudflare)

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Run the pipeline**

```bash
# Step 1 — fetch raw data from FBref (~5 min, opens a Chrome window)
python data_pipeline.py

# Step 2 — engineer features and write the parquet
python prepare_features.py

# Step 3 — launch the dashboard
streamlit run app.py
```

Re-running `data_pipeline.py` uses cached HTML files from the first run for already-fetched pages, so subsequent runs are fast. Delete `.fbref_cache/` to force a full re-fetch.

---

## Tech stack

| | |
|---|---|
| **Data** | [soccerdata](https://github.com/probberechts/soccerdata), pandas, lxml |
| **Model** | scikit-learn (`StandardScaler`, cosine similarity via `NearestNeighbors`) |
| **Dashboard** | Streamlit, matplotlib |
| **Scraping** | SeleniumBase (via soccerdata), lxml HTML parser |

---

## Known limitations

- **Bundesliga**: soccerdata's `"Big 5 European Leagues Combined"` league key silently drops the Bundesliga league label, leaving those rows with a null `league` field. The pipeline uses explicit per-league keys to avoid this, but the Big 5 passing/possession/defense scrape uses the combined URL which includes all five leagues correctly labelled via the Comp column.
- **Mid-season transfers**: players who moved clubs mid-season appear with their stats summed across both spells. The `team` field shows both clubs separated by ` / ` (e.g. `Tottenham / Atlético Madrid`). The `league` field reflects the club where the player spent the most minutes.
- **Enriched features on first run**: if `data_pipeline.py` fails to scrape any of the three secondary tables (passing, possession, defense), `prepare_features.py` will warn and fall back to the 9-feature core set automatically. The dashboard works in either mode.
- **FBref rate limits**: FBref aggressively rate-limits scrapers. The pipeline sleeps between requests and caches responses. If you hit repeated 403s, wait a few hours before re-running.