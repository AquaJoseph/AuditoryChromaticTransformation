import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
NRC_PATH = ROOT / "datasets" / "nrc" / "NRC-VAD-Lexicon-v2.1.txt"
EMO_COL_PATH = ROOT / "datasets" / "emo-col" / "emotions_images_colors.csv"

df_nrc = pd.read_csv(NRC_PATH, sep="\t")
df_emo = pd.read_csv(EMO_COL_PATH)

df_nrc.columns = [c.strip().lower() for c in df_nrc.columns]
df_emo.columns = [c.strip().lower() for c in df_emo.columns]

df_nrc["term"] = df_nrc["term"].str.strip().str.lower()
df_emo["emotion"] = df_emo["emotion"].str.strip().str.lower()

df_emo = df_emo[["emotion", "url", "color"]]

common = set(df_nrc["term"]) & set(df_emo["emotion"])
print(f"Matches found: {len(common)}")

df_nrc_matched = df_nrc[df_nrc["term"].isin(common)].copy()
df_emo_matched = df_emo[df_emo["emotion"].isin(common)].copy()

merged = pd.merge(
    df_emo_matched,
    df_nrc_matched,
    left_on="emotion",
    right_on="term",
    how="inner"
)

def normalize_vad(x):
    return (x + 1.0) / 2.0

def parse_rgb(rgb_value):
    if pd.isna(rgb_value):
        return None

    if not isinstance(rgb_value, str):
        rgb_value = str(rgb_value)

    rgb_value = rgb_value.strip().strip("()")
    parts = [x.strip() for x in rgb_value.split(",")]

    if len(parts) != 3:
        return None

    try:
        r, g, b = [int(x) for x in parts]
    except ValueError:
        return None

    return r / 255.0, g / 255.0, b / 255.0

merged["valence"] = normalize_vad(merged["valence"])
merged["arousal"] = normalize_vad(merged["arousal"])
merged["dominance"] = normalize_vad(merged["dominance"])

merged = merged[merged["color"].notna()].copy()
merged["rgb"] = merged["color"].apply(parse_rgb)

merged = merged[merged["rgb"].notna()].copy()
merged[["r", "g", "b"]] = pd.DataFrame(merged["rgb"].tolist(), index=merged.index)

final_df = merged[["emotion", "valence", "arousal", "dominance", "r", "g", "b"]].copy()

SAVE_PATH = ROOT / "datasets" / "nrc-emo-color" / "vad_rgb_dataset.csv"
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

final_df.to_csv(SAVE_PATH, index=False)

