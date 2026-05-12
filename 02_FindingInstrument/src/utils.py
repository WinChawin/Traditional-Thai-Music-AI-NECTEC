"""Shared utilities for FindingInstrument notebooks.

Philosophy: only repeated boilerplate lives here. All analysis logic,
models, and feature engineering live in notebooks for readability.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# ---------- Constants ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_FIG_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_TBL_DIR = PROJECT_ROOT / "results" / "tables"

INSTRUMENTS = ["ขลุ่ย-ปี่", "ฆ้องวงใหญ่", "ซออู้", "ระนาดเอก"]
PIECES = ["กล่อมนารี", "จีนขิมเล็ก", "ตับต้นเพลงฉิ่ง",
          "สาธุการ", "แขกมอญบางช้าง", "โหมโรงมหาฤกษ์"]

LOW_MARK = "ฺ"   # ฺ (พินทุ) — below base char
HIGH_MARK = "ํ"  # ํ (นิคหิต) — above base char

# 22-token vocabulary: 7 note bases × 3 octaves (minus impossible combos) + sustain
NOTE_BASES = ["ด", "ร", "ม", "ฟ", "ซ", "ล", "ท"]
SUSTAIN = "-"

# ลำดับ tokens สำหรับ heatmap (sustain, low-octave, mid, high)
TOKEN_ORDER = ([SUSTAIN] +
               [b + LOW_MARK for b in NOTE_BASES] +
               NOTE_BASES +
               [b + HIGH_MARK for b in NOTE_BASES])

# Pitch mapping สำหรับ contour plot — heptatonic step
BASE_PITCH = {"ด": 0, "ร": 1, "ม": 2, "ฟ": 3, "ซ": 4, "ล": 5, "ท": 6}

# 7 hand-crafted feature column names
FEATURE_COLS = ["sustain_ratio", "low_oct_ratio", "mid_oct_ratio",
                "high_oct_ratio", "lukabad_density", "paren_ratio",
                "transition_entropy"]


# ---------- Data loading ----------

def load_notes(path: Path | str | None = None) -> pd.DataFrame:
    """Load notes.parquet and add canonical `piece` column.

    Returns DataFrame with columns:
        instrument, song_file, title, section, bar_index, row, col,
        cell, tokens, n_tokens, has_paren, piece
    """
    if path is None:
        path = DATA_DIR / "notes.parquet"
    df = pd.read_parquet(path)
    df["piece"] = df["song_file"].apply(piece_of)
    return df


def piece_of(song_file: str) -> str:
    """Map song_file (e.g. 'กล่อมนารี เถา ขลุ่ย.xlsx') to canonical piece name."""
    name = song_file.replace(".xlsx", "")
    for p in PIECES:
        if name.startswith(p):
            return p
    return "unknown"


# ---------- Cross-validation ----------

def make_cv_splits(
    df: pd.DataFrame,
    group_col: str = "piece",
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Leave-one-piece-out CV iterator.

    Yields (held_out_piece, train_indices, test_indices) for each of 6 folds.
    """
    for held_out in PIECES:
        test_mask = df[group_col] == held_out
        if not test_mask.any():
            continue
        train_idx = df.index[~test_mask].to_numpy()
        test_idx = df.index[test_mask].to_numpy()
        yield held_out, train_idx, test_idx


# ---------- Reproducibility ----------

def set_seed(seed: int = 42) -> None:
    """Set seeds for python, numpy, and torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------- Token utilities ----------

def token_octave(token: str) -> str:
    """Return octave label for a token: 'low', 'mid', 'high', or 'sustain'."""
    if token == SUSTAIN:
        return "sustain"
    if LOW_MARK in token:
        return "low"
    if HIGH_MARK in token:
        return "high"
    return "mid"


def token_to_pitch(token: str) -> int | None:
    """Map token to heptatonic step (mid base = 0). Sustain returns None.

    Low octave subtracts 7, high octave adds 7.
    """
    if token == SUSTAIN or token == "":
        return None
    base = token.replace(LOW_MARK, "").replace(HIGH_MARK, "")
    if base not in BASE_PITCH:
        return None
    pitch = BASE_PITCH[base]
    if LOW_MARK in token:
        pitch -= 7
    elif HIGH_MARK in token:
        pitch += 7
    return pitch


# ---------- Window builder ----------

def build_windows(
    df: pd.DataFrame,
    window_size: int = 16,
    stride: int = 8,
) -> pd.DataFrame:
    """Slice each song_file into overlapping bar windows.

    Windows never cross song_file boundary. Returns DataFrame with columns:
        song_file, instrument, piece, start_bar, end_bar,
        tokens (flat list), n_tokens_per_bar (list), has_paren_bars, n_bars
    """
    windows = []
    for song_file, group in df.groupby("song_file", sort=False):
        group = group.sort_values("bar_index").reset_index(drop=True)
        n_bars = len(group)
        for start in range(0, n_bars - window_size + 1, stride):
            end = start + window_size
            sub = group.iloc[start:end]
            windows.append({
                "song_file":         song_file,
                "instrument":        sub["instrument"].iloc[0],
                "piece":             sub["piece"].iloc[0],
                "start_bar":         sub["bar_index"].iloc[0],
                "end_bar":           sub["bar_index"].iloc[-1],
                "tokens":            [t for ts in sub["tokens"] for t in ts],
                "n_tokens_per_bar":  sub["n_tokens"].tolist(),
                "has_paren_bars":    int(sub["has_paren"].sum()),
                "n_bars":            window_size,
            })
    return pd.DataFrame(windows)


# ---------- Hand-crafted features ----------

def compute_features_window(row: pd.Series) -> pd.Series:
    """Compute 7 hand-crafted features for a single window row.

    Returns Series with: sustain_ratio, low_oct_ratio, mid_oct_ratio,
    high_oct_ratio, lukabad_density, paren_ratio, transition_entropy
    """
    all_tokens = row["tokens"]
    total_tokens = len(all_tokens)
    total_bars = row["n_bars"]

    if total_tokens == 0:
        return pd.Series({f: 0.0 for f in FEATURE_COLS})

    n_sustain = sum(1 for t in all_tokens if t == SUSTAIN)
    note_tokens = [t for t in all_tokens if t != SUSTAIN]
    n_notes = len(note_tokens)
    n_low = sum(1 for t in note_tokens if LOW_MARK in t)
    n_high = sum(1 for t in note_tokens if HIGH_MARK in t)
    n_mid = n_notes - n_low - n_high

    n_lukabad = sum(1 for n in row["n_tokens_per_bar"] if n >= 5)
    n_paren = row["has_paren_bars"]

    bigrams = [(all_tokens[i], all_tokens[i+1]) for i in range(len(all_tokens) - 1)]
    bigram_counts = Counter(bigrams)
    total_bigrams = sum(bigram_counts.values())
    entropy = 0.0
    if total_bigrams > 0:
        for cnt in bigram_counts.values():
            p = cnt / total_bigrams
            entropy -= p * math.log2(p)

    return pd.Series({
        "sustain_ratio":      n_sustain / total_tokens,
        "low_oct_ratio":      n_low / n_notes if n_notes > 0 else 0.0,
        "mid_oct_ratio":      n_mid / n_notes if n_notes > 0 else 0.0,
        "high_oct_ratio":     n_high / n_notes if n_notes > 0 else 0.0,
        "lukabad_density":    n_lukabad / total_bars,
        "paren_ratio":        n_paren / total_bars,
        "transition_entropy": entropy,
    })


def compute_features_dataframe(windows_df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for all windows. Returns dataframe with metadata + 7 features."""
    feat = windows_df.apply(compute_features_window, axis=1)
    meta = windows_df[["song_file", "instrument", "piece", "start_bar", "end_bar"]]
    return pd.concat([meta.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)


# ---------- Output helpers ----------

def save_fig(name: str, fig=None, dpi: int = 150, timestamp: bool = False) -> Path:
    """Save matplotlib figure to results/figures/."""
    import matplotlib.pyplot as plt
    RESULTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    if timestamp:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{name}_{stamp}"
    path = RESULTS_FIG_DIR / f"{name}.png"
    if fig is None:
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_table(df: pd.DataFrame, name: str, timestamp: bool = False) -> Path:
    """Save DataFrame as CSV to results/tables/."""
    RESULTS_TBL_DIR.mkdir(parents=True, exist_ok=True)
    if timestamp:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{name}_{stamp}"
    path = RESULTS_TBL_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path