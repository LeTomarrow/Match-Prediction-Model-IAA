#!/usr/bin/env python3
"""Supplementary pre-match features from raw CSVs (no external data).

Features:
  - Elo ratings (home_elo, away_elo, elo_diff, home_elo_win_prob)
  - Streak features (win/loss/draw/unbeaten streak per club)
  - Days rest and rest advantage
  - League position (from club_games.own_position)
  - Competition rolling averages (league avg goals, strictly pre-match)
  - Formation and attendance from games.csv
"""
from __future__ import annotations

from math import exp, lgamma, log
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# ── Elo constants ─────────────────────────────────────────────────────────────
ELO_K = 30
ELO_HOME_ADV = 100      # points added to home rating when computing expected score
ELO_INIT = 1500.0


def _elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def compute_elo_features(games: pd.DataFrame) -> pd.DataFrame:
    """Pre-match Elo for every game. Uses ALL history (no 2018 cutoff) for warm-up."""
    df = games[["game_id", "date", "home_club_id", "away_club_id",
                "home_club_goals", "away_club_goals"]].copy()
    df["home_club_goals"] = pd.to_numeric(df["home_club_goals"], errors="coerce")
    df["away_club_goals"] = pd.to_numeric(df["away_club_goals"], errors="coerce")
    df = df.dropna(subset=["home_club_id", "away_club_id",
                            "home_club_goals", "away_club_goals"])
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)

    elo: Dict[int, float] = {}
    rows = []

    for row in df.itertuples(index=False):
        h_id = int(row.home_club_id)
        a_id = int(row.away_club_id)
        h_elo = elo.get(h_id, ELO_INIT)
        a_elo = elo.get(a_id, ELO_INIT)

        h_exp = _elo_expected(h_elo + ELO_HOME_ADV, a_elo)
        a_exp = 1.0 - h_exp

        rows.append({
            "game_id": row.game_id,
            "home_elo": h_elo,
            "away_elo": a_elo,
            "elo_diff": h_elo - a_elo,
            "home_elo_win_prob": h_exp,
        })

        hg, ag = int(row.home_club_goals), int(row.away_club_goals)
        h_score = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        a_score = 1.0 - h_score

        elo[h_id] = h_elo + ELO_K * (h_score - h_exp)
        elo[a_id] = a_elo + ELO_K * (a_score - a_exp)

    return pd.DataFrame(rows)


def compute_streak_rest_position(games: pd.DataFrame, club_games: pd.DataFrame) -> pd.DataFrame:
    """Win/loss/draw/unbeaten streaks, days rest, and league position per club per game."""
    cg = club_games[["game_id", "club_id", "own_goals", "opponent_goals",
                      "is_win", "own_position"]].copy()
    date_map = games[["game_id", "date"]].copy()
    cg = cg.merge(date_map, on="game_id", how="inner")
    cg["date"] = pd.to_datetime(cg["date"], errors="coerce")
    cg["is_draw"] = (cg["own_goals"] == cg["opponent_goals"]).astype(int)
    cg = cg.sort_values(["club_id", "date", "game_id"]).reset_index(drop=True)

    feat_rows = []
    for club_id, grp in cg.groupby("club_id", sort=False):
        w_streak = l_streak = d_streak = unbeaten = 0
        last_date = None
        last_pos = np.nan

        for row in grp.itertuples(index=False):
            days_rest = (row.date - last_date).days if last_date is not None else np.nan
            feat_rows.append({
                "game_id": row.game_id,
                "club_id": club_id,
                "win_streak": w_streak,
                "loss_streak": l_streak,
                "draw_streak": d_streak,
                "unbeaten_streak": unbeaten,
                "days_rest": days_rest,
                "league_position": last_pos,
            })
            # update streaks
            if row.is_win:
                w_streak += 1; l_streak = 0; d_streak = 0; unbeaten += 1
            elif row.is_draw:
                w_streak = 0; d_streak += 1; l_streak = 0; unbeaten += 1
            else:
                w_streak = 0; d_streak = 0; l_streak += 1; unbeaten = 0
            last_date = row.date
            if not pd.isna(row.own_position):
                last_pos = float(row.own_position)

    club_df = pd.DataFrame(feat_rows)

    g = games[["game_id", "home_club_id", "away_club_id"]].copy()

    home_f = club_df.rename(columns={
        "club_id": "home_club_id",
        "win_streak": "home_win_streak", "loss_streak": "home_loss_streak",
        "draw_streak": "home_draw_streak", "unbeaten_streak": "home_unbeaten_streak",
        "days_rest": "home_days_rest", "league_position": "home_league_position",
    })
    away_f = club_df.rename(columns={
        "club_id": "away_club_id",
        "win_streak": "away_win_streak", "loss_streak": "away_loss_streak",
        "draw_streak": "away_draw_streak", "unbeaten_streak": "away_unbeaten_streak",
        "days_rest": "away_days_rest", "league_position": "away_league_position",
    })

    out = g.merge(home_f, on=["game_id", "home_club_id"], how="left")
    out = out.merge(away_f, on=["game_id", "away_club_id"], how="left")

    out["rest_advantage"] = out["home_days_rest"] - out["away_days_rest"]
    out["position_diff"] = out["home_league_position"] - out["away_league_position"]
    out["home_short_rest"] = (out["home_days_rest"] < 4).astype(float)
    out["away_short_rest"] = (out["away_days_rest"] < 4).astype(float)

    keep = ["game_id",
            "home_win_streak", "home_loss_streak", "home_draw_streak", "home_unbeaten_streak",
            "home_days_rest", "home_league_position",
            "away_win_streak", "away_loss_streak", "away_draw_streak", "away_unbeaten_streak",
            "away_days_rest", "away_league_position",
            "rest_advantage", "position_diff", "home_short_rest", "away_short_rest"]
    return out[[c for c in keep if c in out.columns]]


def compute_competition_context(games: pd.DataFrame) -> pd.DataFrame:
    """Rolling league-average goals per competition/season (strictly pre-match).
    Also extracts formation, attendance, and competition_type."""
    cols = ["game_id", "date", "competition_id", "season",
            "home_club_goals", "away_club_goals",
            "home_club_formation", "away_club_formation",
            "attendance", "competition_type"]
    df = games[[c for c in cols if c in games.columns]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_club_goals"] = pd.to_numeric(df["home_club_goals"], errors="coerce")
    df["away_club_goals"] = pd.to_numeric(df["away_club_goals"], errors="coerce")
    df["attendance"] = pd.to_numeric(df.get("attendance"), errors="coerce")
    df = df.sort_values(["competition_id", "season", "date", "game_id"]).reset_index(drop=True)

    rows = []
    for (comp, season), grp in df.groupby(["competition_id", "season"], sort=False):
        cum_h = cum_a = n = 0.0
        for row in grp.itertuples(index=False):
            rows.append({
                "game_id": row.game_id,
                "league_avg_home_goals": cum_h / n if n > 0 else np.nan,
                "league_avg_away_goals": cum_a / n if n > 0 else np.nan,
                "league_avg_total_goals": (cum_h + cum_a) / n if n > 0 else np.nan,
                "competition_type_feat": getattr(row, "competition_type", np.nan),
                "home_formation": getattr(row, "home_club_formation", np.nan),
                "away_formation": getattr(row, "away_club_formation", np.nan),
                "attendance": getattr(row, "attendance", np.nan),
            })
            if not (pd.isna(row.home_club_goals) or pd.isna(row.away_club_goals)):
                cum_h += row.home_club_goals
                cum_a += row.away_club_goals
                n += 1

    return pd.DataFrame(rows)


def build_extra_features(data_dir: Path) -> pd.DataFrame:
    """Load raw CSVs and return one-row-per-game_id extra feature table."""
    print("  [extra] Loading games.csv ...")
    games = pd.read_csv(data_dir / "games.csv")
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.dropna(subset=["date"]).copy()

    print("  [extra] Loading club_games.csv ...")
    club_games = pd.read_csv(
        data_dir / "club_games.csv",
        usecols=["game_id", "club_id", "own_goals", "opponent_goals",
                 "is_win", "own_position"],
    )

    print("  [extra] Computing Elo ratings ...")
    elo_df = compute_elo_features(games)

    print("  [extra] Computing streak / rest / position ...")
    streak_df = compute_streak_rest_position(games, club_games)

    print("  [extra] Computing competition context ...")
    comp_df = compute_competition_context(games)

    out = elo_df.merge(streak_df, on="game_id", how="left")
    out = out.merge(comp_df, on="game_id", how="left")
    print(f"  [extra] Extra features: {len(out):,} rows × {len(out.columns)} cols")
    return out
