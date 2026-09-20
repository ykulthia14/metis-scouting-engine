import os
import time
import hashlib
from curl_cffi import requests
import soccerdata as sd
import pandas as pd
from io import StringIO
from lxml import etree, html as lxml_html
from pathlib import Path


SEASONS = ["24-25", "25-26"]
OUTPUT_FILE = "top5_raw_2024_2026.csv"
CACHE_DIR = Path("./.fbref_cache")
RATE_LIMIT_SECS = 4

LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "FRA-Ligue 1",
    "GER-Bundesliga",
    "ITA-Serie A"
]

BIG5_SEASON_MAP = {"24-25": "2024-2025", "25-26": "2025-2026"}
BIG5_BASE = "https://fbref.com/en/comps/Big5/{fbref_season}/{stat}/players/{fbref_season}-Big-5-European-Leagues-Stats"

COMP_TO_LEAGUE = {
    "Premier League": "ENG-Premier League",
    "La Liga": "ESP-La Liga",
    "Ligue 1": "FRA-Ligue 1",
    "Bundesliga": "GER-Bundesliga",
    "Fußball-Bundesliga": "GER-Bundesliga",
    "Serie A": "ITA-Serie A"
}

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ────────────────────────────────────

def _cache_path(url: str) -> Path:
    return CACHE_DIR / (hashlib.md5(url.encode()).hexdigest() + ".html")

def _fetch_html(url: str) -> bytes:
    path = _cache_path(url)
    if path.exists():
        print(f"  (cache hit)")
        return path.read_bytes()
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://fbref.com/",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1"
    }
    resp = requests.get(url, impersonate="chrome120", headers=headers, timeout=30)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    time.sleep(RATE_LIMIT_SECS)
    return resp.content

def _to_canonical_season(s) -> str:
    """
    Normalise any season representation to soccerdata's MULTI_YEAR format:
        "24-25"      → "2425"
        "2024-2025"  → "2425"
        2425         → "2425"
        "2425"       → "2425"
 
    Uses soccerdata's own SeasonCode codec so the output is guaranteed to
    match whatever soccerdata puts in its DataFrame index after reset_index().
    """
    from soccerdata._common import SeasonCode
    return SeasonCode.MULTI_YEAR.parse(str(s))

def _flatten_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from soccerdata into snake_case strings."""
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [str(c) for c in col if c and not str(c).startswith("Unnamed")]
            name = "_".join(parts) if parts else str(col[-1])
        else:
            name = str(col)
        new_cols.append(name.strip())
    df = df.copy()
    df.columns = new_cols
    return df

def _extract_big5_table(html_bytes: bytes, stat: str) -> pd.DataFrame:
    """Extract and parse the primary table or HTML-commented table from FBref."""

    tree = lxml_html.fromstring(html_bytes)
    table_id = f"stats_{stat}"

    tables = tree.xpath(f"//table[@id='{table_id}']")

    if not tables:
        comments = tree.xpath(
            f"//comment()[contains(., \"id='{table_id}'\") or contains(., 'id=\"{table_id}\"')]"
        )
        for c in comments:
            inner_text = str(c)[4:-3]
            parser = etree.HTMLParser(recover=True)
            inner = etree.fromstring(inner_text.encode(), parser)
            found = inner.xpath(f"//table[@id='{table_id}']")
            if found:
                tables = [lxml_html.fromstring(etree.tostring(found[0]).decode())]
                break

    if not tables:
        raise ValueError(f"Table '{table_id}' not found on page")

    df = pd.read_html(StringIO(lxml_html.tostring(tables[0]).decode()), header=[0, 1])[0]

    rk_col = df.columns[0]  # Drop repeated header rows (FBref repeats headers every N rows)
    df = df[df[rk_col].astype(str) != "Rk"].copy()

    df.columns = [
        "_".join(str(c) for c in col if str(c) and not str(c).startswith("Unnamed")).strip("_")  
        for col in df.columns 
    ]  # Flatten the two-row header

    return df

# ── step 1: standard / shooting / misc via soccerdata ───────────

def fetch_soccerdata_tables() -> pd.DataFrame:
    print("Fetching standard / shooting / misc via soccerdata...")
    fbref = sd.FBref(leagues=LEAGUES, seasons=SEASONS)

    dfs = {}

    for stat in ["standard", "shooting", "misc"]:
        print(f"  Fetching '{stat}'...")
        df = fbref.read_player_season_stats(stat_type=stat)
        dfs[stat] = _flatten_multiindex(df.reset_index())

    # Normalise season to "2425" format before any merging or joining.
    # soccerdata already outputs "2425" for MULTI_YEAR leagues, but we run it
    # through the codec explicitly so any edge-case format is covered.
    for name in dfs:
        if "season" in dfs[name].columns:
            dfs[name]["season"] = dfs[name]["season"].map(_to_canonical_season)


    base = dfs["standard"]
    keys = [k for k in ["league", "season", "team", "player"] if k in base.columns]

    # Explicitly request the most important columns from each table to avoid
    # accidentally deduplicating them (e.g. Expected_xG from shooting).

    SHOOTING_WANT = ["Expected_xG", "Expected_npxG", "Expected_xAG", "Standard_Sh", "Standard_SoT"]
    MISC_WANT = ["Performance_TklW", "Performance_Int", "Performance_Crs", "Performance_Fld", "Performance_Fls", "Performance_2CrdY", "Performance_OG", "Performance_Off"]

    for stat, want in [("shooting", SHOOTING_WANT), ("misc", MISC_WANT)]:
        other = dfs[stat]
        explicit  = [c for c in want if c in other.columns and c not in base.columns]
        fallback  = [c for c in other.columns if c not in base.columns and c not in explicit]
        extra_cols = explicit + fallback
        if extra_cols:
            merge_keys = [k for k in keys if k in other.columns]
            base = base.merge(other[merge_keys + extra_cols], on=merge_keys, how="left")
    print(f" soccerdata tables merged: {len(base)} rows, {len(base.columns)} cols")
    return base


# ── step 2: passing / possession / defense scraped directly ────────
def _get_big5_url(stat: str, fbref_season: str) -> str:
    # URL structure includes season slug in directory and filename
    return f"https://fbref.com/en/comps/Big5/{fbref_season}/{stat}/players/{fbref_season}-Big-5-European-Leagues-Stats"

def _scrape_big5_stat(stat: str, fbref_season: str, sd_season: str) -> pd.DataFrame:
    url = _get_big5_url(stat, fbref_season)
    print(f" Scraping {stat} / {fbref_season} → {url}")
    html_bytes = _fetch_html(url)
    df = _extract_big5_table(html_bytes, stat)

    comp_col = next((c for c in df.columns if c.lower() in ("comp", "competition")), None)
    if comp_col:
        df["league"] = df[comp_col].map(COMP_TO_LEAGUE).fillna(df[comp_col])
        df = df.drop(columns=[comp_col])
    else:
        df["league"] = "Unknown"

    df["season"] = _to_canonical_season(sd_season)

    # Normalise the Player column name
    player_col = next((c for c in df.columns if c.lower() in ("player", "player_player")), None)
    if player_col and player_col != "player":
        df = df.rename(columns={player_col: "player"})

    squad_col = next((c for c in df.columns if c.lower() in ("squad", "squad_squad", "team")), None)
    if squad_col and squad_col != "team":
        df = df.rename(columns={squad_col: "team"})

    df= df[df["player"].notna() & (df["player"].astype(str).str.strip() != "")]
    return df

def fetch_secondary_tables() -> dict[str, pd.DataFrame]:
    """Returns {"passing": df, "possession": df, "defense": df} combined across both seasons."""
    print("Scraping passing / possession / defense from FBref Big 5 pages...")
    frames: dict[str, list] = {"passing": [], "possession": [], "defense": []}

    for sd_season, fbref_season in BIG5_SEASON_MAP.items():
        if sd_season not in SEASONS:
            continue
        for stat in frames:
            try:
                df = _scrape_big5_stat(stat, fbref_season, sd_season)
                frames[stat].append(df)
                print(f" ✓ {stat} / {sd_season}: {len(df)} rows")
            except Exception as e:
                print(f" ✗ WARNING: {stat} / {sd_season} failed — {e}")
                print(f" Continuing without it; re-run or add manually if needed.")

    return {stat: pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            for stat, dfs in frames.items()}


# ── step 3: extract relevant columns and merge ─────────────
# Map from FBref's two-row-header column names (as flattened by _extract_big5_table)
# to our clean internal names. Add more here if you want additional features.

PASSING_COLS = {
    "Expected_xAG": "xag",
    "KP": "key_passes",
    "1/3": "passes_final_third",
    "PPA": "passes_penalty_area",
    "PrgP": "progressive_passes",
    "Cmp": "passes_completed",
    "Att": "passes_attempted",
    "Passing_KP": "key_passes",
    "Passing_PrgP": "progressive_passes",
    "Passing_Cmp": "passes_completed",
    "Passing_Att": "passes_attempted",
    "Expected_xA": "xa",
}

POSSESSION_COLS = {
    "Take-Ons_Att": "dribbles_attempted",
    "Take-Ons_Succ": "dribbles_completed",
    "Carries_PrgC": "progressive_carries",
    "Carries_TotDist": "carry_distance",
    "Carries_PrgDist": "carry_progressive_distance",
    "Take-Ons_Att%": None,
    "Att": "dribbles_attempted",
    "Succ": "dribbles_completed",
    "PrgC": "progressive_carries",
}

DEFENSE_COLS = {
    "Pressures_Press": "pressures",
    "Pressures_Succ": "pressures_successful",
    "Tackles_Tkl": "tackles",
    "Int": "interceptions",
    "Blocks_Blocks": "blocks",
    "Tackles_TklW": "tackles_won",
    "Press": "pressures",
    "Tkl": "tackles",
}

def _select_and_rename(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Pick columns from df that appear in col_map, rename them, drop None-mapped ones."""
    result = {}
    for src, dst in col_map.items():
        if dst is None:
            continue
        if src in df.columns:
            result[dst] = df[src]
    return pd.DataFrame(result, index=df.index)

def merge_all(base: pd.DataFrame, secondary: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Join on player + team + season only.
    # Excluding league avoids any Comp-column mapping mismatch in the scraped tables
    # (where a single wrong league label would silently NaN every row for that league).
    # player + team + season is uniquely identifying for our purposes.
    join_keys = ["player", "team", "season"]

    def _safe_merge(left: pd.DataFrame, right: pd.DataFrame, col_map: dict, label: str) -> pd.DataFrame:
        if right.empty:
            print(f"  Skipping {label} merge — no data scraped")
            return left

        # Print all scraped columns and league values for diagnosis
        non_key_cols = [c for c in right.columns if c not in join_keys]
        print(f"  {label} scraped columns ({len(non_key_cols)}): {non_key_cols}")
        if "league" in right.columns:
            print(f"  {label} scraped leagues: {right['league'].value_counts(dropna=False).to_dict()}")

        feature_df = _select_and_rename(right, col_map)
        if feature_df.empty:
            print(f"  WARNING: {label} — none of the expected columns found in scraped table.")
            return left

        for k in join_keys:
            if k in right.columns:
                feature_df[k] = right[k].values

        available_keys = [k for k in join_keys if k in left.columns and k in feature_df.columns]
        new_cols = [c for c in feature_df.columns if c not in available_keys and c not in left.columns]

        if not new_cols:
            print(f" WARNING: {label} — no new columns to add after deduplication")
            return left

        merged = left.merge(feature_df[available_keys + new_cols], on=available_keys, how="left")
        gained = [c for c in new_cols if c in merged.columns]
        print(f" Merged {label}: +{len(gained)} columns ({gained})")
        return merged

    result = base
    result = _safe_merge(result, secondary["passing"], PASSING_COLS, "passing")
    result = _safe_merge(result, secondary["possession"], POSSESSION_COLS, "possession")
    result = _safe_merge(result, secondary["defense"], DEFENSE_COLS, "defense")
    return result


# ── step 4: clean and save ────────────────
def clean_and_save(df: pd.DataFrame) -> None:
    def season_to_int(s):
        try:
            return int(str(s).replace("-", ""))
        except Exception:
            return s

    df["season"] = df["season"].apply(season_to_int)

    # Ensure minutes column is numeric and filter out zero-minutes rows
    min_col = next((c for c in df.columns if "Min" in c and "Playing" in c), None)
    if min_col:
        df[min_col] = pd.to_numeric(df[min_col], errors="coerce").fillna(0)
        df = df[df[min_col] > 0].copy()
    df = df.reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved {len(df)} rows, {len(df.columns)} columns → '{OUTPUT_FILE}'")
    print(f" Columns: {list(df.columns)}")


# ── main ──────────────────────────
if __name__ == "__main__":
    base = fetch_soccerdata_tables()
    secondary = fetch_secondary_tables()
    merged = merge_all(base, secondary)
    clean_and_save(merged)