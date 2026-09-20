import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "top5_raw_2024_2026.csv"
OUTPUT_FILE = "scouting_features.parquet"
MIN_MINUTES = 600

# ── Load & coerce ───────────────────────────
print("Loading raw player stats...")
df = pd.read_csv(INPUT_FILE)

df.columns = [c.strip() for c in df.columns]

# Coerce every numeric-ish column up front to avoid silent string maths later
skip = {"league", "season", "team", "player", "nation", "pos", "age", "born"}
for col in df.columns:
    if col not in skip:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)


# ── Collapse mid-season transfers ──────────────────────
# Sort so the highest-minutes club comes first in each player-season group.
# meta columns take the "first" value (= primary club); numeric columns are summed.

num_cols = [c for c in df.columns if c not in skip and c != "season"]
meta_solo = [c for c in ["league", "pos", "age", "nation", "born"] if c in df.columns]

agg_rules = {c: "first" for c in meta_solo}
agg_rules["team"] = lambda s: " / ".join(dict.fromkeys(s.astype(str)))
agg_rules.update({c: "sum" for c in num_cols})

df = df.sort_values(["player", "season", "Playing Time_Min"], ascending=[True, True, False])

print(f"Rows before collapsing transfers : {len(df)}")
df = df.groupby(["player", "season"], as_index=False).agg(agg_rules)
print(f"Rows after collapsing transfers  : {len(df)}")

# ── Minute filter & goalkeeper removal ──────────
df = df[df["Playing Time_Min"] >= MIN_MINUTES].copy()
df = df[~df["pos"].astype(str).str.contains("GK", na=False)].copy()
df["primary_pos"] = df["pos"].apply(lambda x: x.split(",")[0].strip() if pd.notna(x) else "Unknown")

# ── Per-90 calculation ───────────────────
n90 = np.maximum(np.where(
    df["Playing Time_90s"] > 0,
    df["Playing Time_90s"],
    df["Playing Time_Min"] / 90.0
), 0.01)

def p90(col: str) -> pd.Series:
    """Divide a raw counting stat by 90s played. Returns 0 if column missing."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return (df[col] / n90).round(3)

def safe_ratio(num: str, den: str) -> pd.Series:
    """num / den, 0 where denominator is 0 or missing."""
    n = df.get(num, pd.Series(0.0, index=df.index))
    d = df.get(den, pd.Series(0.0, index=df.index))
    return np.where(d > 0, (n / d).round(3), 0.0)


# — Attack —
df["goals_np_p90"] = p90("Performance_G-PK")
# xG: soccerdata shooting table produces "expected_xG" (snake_cased level-0 header)
_xg_col = next((c for c in ["expected_xG", "expected_npxG", "xg", "Expected_xG", "Expected_npxG"]
                if c in df.columns), None)
df["xg_p90"] = p90(_xg_col) if _xg_col else pd.Series(0.0, index=df.index)
df["shots_p90"] = p90("Standard_Sh")
df["shots_on_target_p90"] = p90("Standard_SoT")
df["shot_accuracy"] = safe_ratio("Standard_SoT", "Standard_Sh")

# — Creation —
df["assists_p90"] = p90("Performance_Ast")
# xAG: soccerdata shooting → "expected_xAG"; Big5 passing scrape → "xag" (via col_map rename)
_xag_col = next((c for c in ["xag", "expected_xAG", "Expected_xAG"] if c in df.columns), None)
df["xag_p90"] = p90(_xag_col) if _xag_col else pd.Series(0.0, index=df.index)
df["key_passes_p90"] = p90("key_passes") if "key_passes" in df.columns else pd.Series(0.0, index=df.index)
df["passes_final_third_p90"] = p90("passes_final_third") if "passes_final_third" in df.columns else pd.Series(0.0, index=df.index)

# — Progression —
df["progressive_passes_p90"] = p90("progressive_passes") if "progressive_passes" in df.columns else pd.Series(0.0, index=df.index)
df["progressive_carries_p90"] = p90("progressive_carries") if "progressive_carries" in df.columns else pd.Series(0.0, index=df.index)
df["pass_completion_pct"] = safe_ratio("passes_completed", "passes_attempted")

# — Dribbling —
df["dribbles_p90"] = p90("dribbles_completed") if "dribbles_completed" in df.columns else pd.Series(0.0, index=df.index)
df["dribble_success_pct"] = safe_ratio("dribbles_completed", "dribbles_attempted")

# — Defending —
df["tackles_won_p90"] = p90("Performance_TklW")
df["interceptions_p90"] = p90("Performance_Int")
# pressures: scraped defense table → "pressures"; soccerdata misc table → "performance_Press"
_press_col = next((c for c in ["pressures", "performance_Press"] if c in df.columns), None)
df["pressures_p90"] = p90(_press_col) if _press_col else pd.Series(0.0, index=df.index)
df["blocks_p90"] = p90("blocks") if "blocks" in df.columns else pd.Series(0.0, index=df.index)
df["crosses_p90"] = p90("Performance_Crs")
df["fouls_drawn_p90"] = p90("Performance_Fld")


# ── Feature set ───────────────────────────────────────────────────────────────
# Split into "always available" (from standard/shooting/misc — already in the CSV)
# and "enriched" (from passing/possession/defense — added by the new pipeline).
# If an enriched column is all zeros it means the scrape didn't come through;
# we log a warning and exclude it from the similarity feature set automatically.
CORE_FEATURES = [
    "goals_np_p90",
    "shots_p90", "shots_on_target_p90", "shot_accuracy",
    "assists_p90",
    "tackles_won_p90", "interceptions_p90",
    "crosses_p90", "fouls_drawn_p90",
]

# xg_p90 and xag_p90 come from the shooting table — present after data_pipeline.py
# runs with FBref access. Listed first so they join the model as soon as they exist.
ENRICHED_FEATURES = [
    "xg_p90",
    "xag_p90",
    "key_passes_p90",
    "passes_final_third_p90",
    "progressive_passes_p90",
    "progressive_carries_p90",
    "pass_completion_pct",
    "dribbles_p90",
    "dribble_success_pct",
    "pressures_p90",
    "blocks_p90",
]

# Include enriched features only if they have meaningful data (>5% non-zero)
active_enriched = []
for feat in ENRICHED_FEATURES:
    if feat in df.columns:
        nonzero = (df[feat] != 0).mean()
        if nonzero >= 0.05:
            active_enriched.append(feat)
        else:
            print(f"  WARN: '{feat}' is {nonzero:.1%} non-zero — excluded from similarity model "
                  f"(re-run data_pipeline.py to populate it)")

FEATURE_COLS = CORE_FEATURES + active_enriched
print(f"\nFeature set ({len(FEATURE_COLS)} features):")
for f in FEATURE_COLS:
    print(f"  {'✓' if f in active_enriched else '·'} {f}")


# ── Per-position scaling ────────

scaled_dfs = []
scalers    = {}

for pos_group, group_df in df.groupby("primary_pos"):
    scalers[pos_group] = StandardScaler()
    scaled_vals = scalers[pos_group].fit_transform(group_df[FEATURE_COLS].fillna(0))
    scaled_group = pd.DataFrame(
        scaled_vals,
        columns=[f"scaled_{col}" for col in FEATURE_COLS],
        index=group_df.index
    )
    scaled_dfs.append(scaled_group)

scaled_all = pd.concat(scaled_dfs).sort_index()

# ── Save ──────────────────
META_COLS = ["player", "team", "league", "season", "pos", "primary_pos", "age", "Playing Time_Min", "Playing Time_90s"]
final_df = pd.concat([df[META_COLS + FEATURE_COLS].reset_index(drop=True), scaled_all.reset_index(drop=True)], axis=1)
final_df.to_parquet(OUTPUT_FILE, index=False)
print(f"\n✓ Saved {len(final_df)} players × {len(final_df.columns)} columns → '{OUTPUT_FILE}'")


# Quick sanity check
print("\nScaling integrity (mean≈0, std≈1 per position):")
scaled_cols = [c for c in final_df.columns if c.startswith("scaled_")]
for pos, g in final_df.groupby("primary_pos"):
    m = g[scaled_cols].mean().mean()
    s = g[scaled_cols].std().mean()
    print(f"  {pos:3s}: mean={m:+.4f}  std={s:.4f}  n={len(g)}")

print("\nFeature coverage (% non-zero):")
for f in FEATURE_COLS:
    pct = (final_df[f] != 0).mean()
    flag = "" if pct >= 0.05 else "  ← LOW"
    print(f"  {f:<35s} {pct:5.1%}{flag}")