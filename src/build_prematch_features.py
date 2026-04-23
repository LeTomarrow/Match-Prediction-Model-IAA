#!/usr/bin/env python3
"""Build pre-match match-level features from raw Transfermarkt-style tables.

The script creates one row per game, using only information available before
that game's date. It builds separate home/away team blocks and joins them with
head-to-head and context features.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROLL_WINDOWS = (3, 5, 10)
FORM_WINDOWS = (3, 5)
TRANSFER_WINDOWS_DAYS = (180, 365)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pre-match feature table")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory containing CSV inputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "match_features.csv",
        help="Path for output CSV",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional limit for number of games (sorted by date) for faster dry runs",
    )
    parser.add_argument(
        "--feature-set",
        choices=["requested", "full"],
        default="requested",
        help="Output schema: 'requested' keeps only requested feature groups, 'full' keeps all columns",
    )
    return parser.parse_args()


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def parse_currency_to_eur(value: object) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan

    s = s.replace("€", "").replace(",", "")
    negative = s.startswith("-")
    s = s.replace("+", "").replace("-", "")

    multiplier = 1.0
    if s.endswith("bn"):
        multiplier = 1e9
        s = s[:-2]
    elif s.endswith("m"):
        multiplier = 1e6
        s = s[:-1]
    elif s.endswith("k"):
        multiplier = 1e3
        s = s[:-1]

    try:
        val = float(s) * multiplier
    except ValueError:
        return np.nan
    return -val if negative else val


def weighted_recent_mean(values: np.ndarray) -> float:
    n = len(values)
    if n == 0:
        return np.nan
    w = np.arange(1, n + 1, dtype=float)
    return float(np.dot(values, w) / w.sum())


def build_games_label_table(data_dir: Path, max_games: int | None = None) -> pd.DataFrame:
    games = pd.read_csv(data_dir / "games.csv")
    keep = [
        "game_id",
        "competition_id",
        "season",
        "round",
        "date",
        "home_club_id",
        "away_club_id",
        "home_club_goals",
        "away_club_goals",
    ]
    games = games[keep].copy()
    games["date"] = safe_to_datetime(games["date"])
    games = games.dropna(subset=["date", "home_club_id", "away_club_id"])\
                 .sort_values(["date", "game_id"])\
                 .reset_index(drop=True)
    if max_games is not None and max_games > 0:
        games = games.head(max_games).copy()
    return games


def build_club_match_base(games: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    club_games = pd.read_csv(data_dir / "club_games.csv")
    cols = [
        "game_id",
        "club_id",
        "own_goals",
        "opponent_goals",
        "opponent_id",
        "hosting",
        "is_win",
    ]
    club_games = club_games[cols].copy()
    base = club_games.merge(games[["game_id", "date"]], on="game_id", how="inner")
    base["is_draw"] = (base["own_goals"] == base["opponent_goals"]).astype(int)
    base["points"] = base["is_win"] * 3 + base["is_draw"]
    base["goal_diff"] = base["own_goals"] - base["opponent_goals"]
    base["clean_sheet"] = (base["opponent_goals"] == 0).astype(int)
    base["hosting"] = base["hosting"].fillna("Unknown")
    base = base.sort_values(["club_id", "date", "game_id"]).reset_index(drop=True)
    return base


def add_rolling_strength_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    group = df.groupby("club_id", group_keys=False)

    shifted_points = group["points"].shift(1)
    shifted_gd = group["goal_diff"].shift(1)
    shifted_gs = group["own_goals"].shift(1)
    shifted_ga = group["opponent_goals"].shift(1)
    shifted_cs = group["clean_sheet"].shift(1)
    shifted_win = group["is_win"].shift(1)

    for w in ROLL_WINDOWS:
        df[f"{prefix}ppg_{w}"] = shifted_points.groupby(df["club_id"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{prefix}goal_diff_avg_{w}"] = shifted_gd.groupby(df["club_id"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{prefix}goals_scored_avg_{w}"] = shifted_gs.groupby(df["club_id"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{prefix}goals_conceded_avg_{w}"] = shifted_ga.groupby(df["club_id"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{prefix}clean_sheet_rate_{w}"] = shifted_cs.groupby(df["club_id"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{prefix}win_rate_{w}"] = shifted_win.groupby(df["club_id"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)

    for w in FORM_WINDOWS:
        df[f"{prefix}form_weighted_points_{w}"] = shifted_points.groupby(df["club_id"]).rolling(w, min_periods=1).apply(weighted_recent_mean, raw=True).reset_index(level=0, drop=True)
        df[f"{prefix}form_weighted_goal_diff_{w}"] = shifted_gd.groupby(df["club_id"]).rolling(w, min_periods=1).apply(weighted_recent_mean, raw=True).reset_index(level=0, drop=True)

    return df


def build_availability_features(club_match: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    appearances = pd.read_csv(
        data_dir / "appearances.csv",
        usecols=["game_id", "player_id", "player_club_id", "minutes_played"],
    )
    appearances = appearances.rename(columns={"player_club_id": "club_id"})
    appearances["minutes_played"] = pd.to_numeric(appearances["minutes_played"], errors="coerce").fillna(0.0)

    lineups = pd.read_csv(
        data_dir / "game_lineups.csv",
        usecols=["game_id", "club_id", "player_id", "type"],
    )
    starters = lineups[lineups["type"] == "starting_lineup"][
        ["game_id", "club_id", "player_id"]
    ].copy()

    starter_sets: Dict[Tuple[int, int], set] = (
        starters.groupby(["club_id", "game_id"])["player_id"].apply(set).to_dict()
    )

    minutes_map: Dict[Tuple[int, int], Dict[int, float]] = {}
    for (club_id, game_id), grp in appearances.groupby(["club_id", "game_id"]):
        minutes_map[(club_id, game_id)] = dict(zip(grp["player_id"], grp["minutes_played"]))

    feats = []
    for club_id, grp in club_match.groupby("club_id", sort=False):
        hist_starter_counts = deque(maxlen=5)
        hist_starter_sets = deque(maxlen=6)
        hist_player_starts = deque(maxlen=5)
        hist_player_minutes = deque(maxlen=5)

        for row in grp.itertuples(index=False, name=None):
            game_id = row[0]
            if hist_starter_counts:
                recent_starter_count = float(np.mean(hist_starter_counts))
            else:
                recent_starter_count = np.nan

            if hist_starter_sets and len(hist_starter_sets) >= 2:
                rots = []
                for prev_set, curr_set in zip(hist_starter_sets, list(hist_starter_sets)[1:]):
                    denom = max(len(prev_set), len(curr_set), 1)
                    overlap = len(prev_set.intersection(curr_set))
                    rots.append(1.0 - overlap / denom)
                rotation_rate = float(np.mean(rots)) if rots else np.nan
            else:
                rotation_rate = np.nan

            if hist_player_starts:
                starts_counter = Counter()
                mins_counter = Counter()
                for c in hist_player_starts:
                    starts_counter.update(c)
                for c in hist_player_minutes:
                    mins_counter.update(c)

                ordered_players = sorted(
                    starts_counter,
                    key=lambda p: (starts_counter[p], mins_counter[p]),
                    reverse=True,
                )
                top11 = ordered_players[:11]

                if top11:
                    hist_n = len(hist_player_starts)
                    typical_share = float(np.mean([starts_counter[p] / hist_n for p in top11]))
                    expected_minutes = float(np.mean([mins_counter[p] for p in top11]))
                else:
                    typical_share = np.nan
                    expected_minutes = np.nan

                heavy_usage_count = int(sum(1 for m in mins_counter.values() if m >= 270.0))
            else:
                typical_share = np.nan
                expected_minutes = np.nan
                heavy_usage_count = np.nan

            feats.append(
                {
                    "game_id": game_id,
                    "club_id": club_id,
                    "availability_typical_starter_share_5": typical_share,
                    "availability_recent_expected_minutes_5": expected_minutes,
                    "availability_heavy_usage_players_5": heavy_usage_count,
                    "availability_recent_starter_count_5": recent_starter_count,
                    "availability_rotation_rate_5": rotation_rate,
                }
            )

            key = (club_id, game_id)
            curr_starters = starter_sets.get(key, set())
            curr_mins = minutes_map.get(key, {})
            curr_start_counts = Counter({p: 1 for p in curr_starters})

            hist_starter_counts.append(len(curr_starters))
            hist_starter_sets.append(curr_starters)
            hist_player_starts.append(curr_start_counts)
            hist_player_minutes.append(Counter(curr_mins))

    return pd.DataFrame(feats)


def build_squad_value_features(games: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    clubs = pd.read_csv(
        data_dir / "clubs.csv",
        usecols=["club_id", "total_market_value", "squad_size", "average_age"],
    )
    clubs["clubs_total_market_value_eur"] = clubs["total_market_value"].apply(parse_currency_to_eur)
    clubs["clubs_squad_size"] = pd.to_numeric(clubs["squad_size"], errors="coerce")
    clubs["clubs_average_age"] = pd.to_numeric(clubs["average_age"], errors="coerce")
    clubs = clubs[["club_id", "clubs_total_market_value_eur", "clubs_squad_size", "clubs_average_age"]]

    valuations = pd.read_csv(
        data_dir / "player_valuations.csv",
        usecols=["player_id", "date", "market_value_in_eur", "current_club_id"],
    )
    valuations = valuations.rename(columns={"current_club_id": "club_id"})
    valuations["date"] = safe_to_datetime(valuations["date"])
    valuations["market_value_in_eur"] = pd.to_numeric(
        valuations["market_value_in_eur"], errors="coerce"
    )
    valuations = valuations.dropna(subset=["date", "club_id", "player_id", "market_value_in_eur"])
    valuations = valuations.sort_values(["club_id", "date", "player_id"]).reset_index(drop=True)

    club_dates = pd.concat(
        [
            games[["date", "home_club_id"]].rename(columns={"home_club_id": "club_id"}),
            games[["date", "away_club_id"]].rename(columns={"away_club_id": "club_id"}),
        ],
        ignore_index=True,
    ).drop_duplicates().sort_values(["club_id", "date"]).reset_index(drop=True)

    out_rows = []
    for club_id, cd_grp in club_dates.groupby("club_id", sort=False):
        v_grp = valuations[valuations["club_id"] == club_id]
        v_grp = v_grp.sort_values("date")

        player_values: Dict[int, float] = {}
        idx = 0
        dates_arr = v_grp["date"].to_numpy()
        players_arr = v_grp["player_id"].to_numpy()
        vals_arr = v_grp["market_value_in_eur"].to_numpy()

        for d in cd_grp["date"].to_list():
            while idx < len(v_grp) and dates_arr[idx] <= d:
                player_values[int(players_arr[idx])] = float(vals_arr[idx])
                idx += 1

            if player_values:
                vals = np.fromiter(player_values.values(), dtype=float)
                total_mv = float(vals.sum())
                avg_mv = float(vals.mean())
                k = min(11, len(vals))
                top11_mv = float(np.partition(vals, len(vals) - k)[-k:].sum())
                value_per_starter = top11_mv / 11.0
            else:
                total_mv = np.nan
                avg_mv = np.nan
                top11_mv = np.nan
                value_per_starter = np.nan

            out_rows.append(
                {
                    "club_id": club_id,
                    "date": d,
                    "squad_total_market_value_eur": total_mv,
                    "squad_avg_market_value_eur": avg_mv,
                    "squad_top11_market_value_eur": top11_mv,
                    "squad_value_per_starter_eur": value_per_starter,
                }
            )

    squad = pd.DataFrame(out_rows).merge(clubs, on="club_id", how="left")

    # Backfill total value from static clubs snapshot if dynamic valuation is missing.
    squad["squad_total_market_value_eur"] = squad["squad_total_market_value_eur"].fillna(
        squad["clubs_total_market_value_eur"]
    )
    return squad


def build_transfer_features(games: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    transfers = pd.read_csv(
        data_dir / "transfers.csv",
        usecols=[
            "transfer_date",
            "from_club_id",
            "to_club_id",
            "transfer_fee",
            "market_value_in_eur",
        ],
    )
    transfers["transfer_date"] = safe_to_datetime(transfers["transfer_date"])
    transfers["transfer_fee"] = pd.to_numeric(transfers["transfer_fee"], errors="coerce").fillna(0.0)
    transfers["market_value_in_eur"] = pd.to_numeric(
        transfers["market_value_in_eur"], errors="coerce"
    ).fillna(0.0)
    transfers = transfers.dropna(subset=["transfer_date"]).copy()

    incoming = transfers[["transfer_date", "to_club_id", "transfer_fee", "market_value_in_eur"]].copy()
    incoming = incoming.rename(columns={"to_club_id": "club_id"})
    incoming["incoming_count"] = 1
    incoming["incoming_fee"] = incoming["transfer_fee"]
    incoming["incoming_mv"] = incoming["market_value_in_eur"]
    incoming = incoming[["transfer_date", "club_id", "incoming_count", "incoming_fee", "incoming_mv"]]

    outgoing = transfers[["transfer_date", "from_club_id", "transfer_fee", "market_value_in_eur"]].copy()
    outgoing = outgoing.rename(columns={"from_club_id": "club_id"})
    outgoing["outgoing_count"] = 1
    outgoing["outgoing_fee"] = outgoing["transfer_fee"]
    outgoing["outgoing_mv"] = outgoing["market_value_in_eur"]
    outgoing = outgoing[["transfer_date", "club_id", "outgoing_count", "outgoing_fee", "outgoing_mv"]]

    transfer_events = (
        incoming.merge(outgoing, on=["transfer_date", "club_id"], how="outer")
        .fillna(0.0)
        .sort_values(["club_id", "transfer_date"])
        .reset_index(drop=True)
    )

    club_dates = pd.concat(
        [
            games[["date", "home_club_id"]].rename(columns={"home_club_id": "club_id"}),
            games[["date", "away_club_id"]].rename(columns={"away_club_id": "club_id"}),
        ],
        ignore_index=True,
    ).drop_duplicates().sort_values(["club_id", "date"]).reset_index(drop=True)

    out_rows = []
    for club_id, cd_grp in club_dates.groupby("club_id", sort=False):
        e = transfer_events[transfer_events["club_id"] == club_id].copy()
        e = e.sort_values("transfer_date")

        if e.empty:
            for d in cd_grp["date"].to_list():
                row = {"club_id": club_id, "date": d}
                for wd in TRANSFER_WINDOWS_DAYS:
                    row[f"transfer_in_count_{wd}d"] = 0.0
                    row[f"transfer_out_count_{wd}d"] = 0.0
                    row[f"transfer_in_fee_{wd}d"] = 0.0
                    row[f"transfer_out_fee_{wd}d"] = 0.0
                    row[f"transfer_net_spend_{wd}d"] = 0.0
                    row[f"transfer_net_incoming_value_{wd}d"] = 0.0
                    row[f"transfer_squad_churn_count_{wd}d"] = 0.0
                out_rows.append(row)
            continue

        dates = e["transfer_date"].to_numpy()
        inc_c = e["incoming_count"].to_numpy(dtype=float)
        out_c = e["outgoing_count"].to_numpy(dtype=float)
        inc_f = e["incoming_fee"].to_numpy(dtype=float)
        out_f = e["outgoing_fee"].to_numpy(dtype=float)
        inc_mv = e["incoming_mv"].to_numpy(dtype=float)
        out_mv = e["outgoing_mv"].to_numpy(dtype=float)

        c_inc_c = np.r_[0.0, np.cumsum(inc_c)]
        c_out_c = np.r_[0.0, np.cumsum(out_c)]
        c_inc_f = np.r_[0.0, np.cumsum(inc_f)]
        c_out_f = np.r_[0.0, np.cumsum(out_f)]
        c_inc_mv = np.r_[0.0, np.cumsum(inc_mv)]
        c_out_mv = np.r_[0.0, np.cumsum(out_mv)]

        for d in cd_grp["date"].to_list():
            row = {"club_id": club_id, "date": d}
            end = np.searchsorted(dates, d, side="left")

            for wd in TRANSFER_WINDOWS_DAYS:
                start_date = d - pd.Timedelta(days=wd)
                start = np.searchsorted(dates, start_date, side="left")

                in_count = c_inc_c[end] - c_inc_c[start]
                out_count = c_out_c[end] - c_out_c[start]
                in_fee = c_inc_f[end] - c_inc_f[start]
                out_fee = c_out_f[end] - c_out_f[start]
                in_mv_sum = c_inc_mv[end] - c_inc_mv[start]
                out_mv_sum = c_out_mv[end] - c_out_mv[start]

                row[f"transfer_in_count_{wd}d"] = in_count
                row[f"transfer_out_count_{wd}d"] = out_count
                row[f"transfer_in_fee_{wd}d"] = in_fee
                row[f"transfer_out_fee_{wd}d"] = out_fee
                row[f"transfer_net_spend_{wd}d"] = in_fee - out_fee
                row[f"transfer_net_incoming_value_{wd}d"] = in_mv_sum - out_mv_sum
                row[f"transfer_squad_churn_count_{wd}d"] = in_count + out_count

            out_rows.append(row)

    return pd.DataFrame(out_rows)


def build_head_to_head_features(games: pd.DataFrame) -> pd.DataFrame:
    df = games[[
        "game_id",
        "date",
        "home_club_id",
        "away_club_id",
        "home_club_goals",
        "away_club_goals",
    ]].copy()
    df["club_min"] = df[["home_club_id", "away_club_id"]].min(axis=1)
    df["club_max"] = df[["home_club_id", "away_club_id"]].max(axis=1)
    df = df.sort_values(["club_min", "club_max", "date", "game_id"]).reset_index(drop=True)

    rows = []
    for (_, _), grp in df.groupby(["club_min", "club_max"], sort=False):
        hist: List[pd.Series] = []
        for row in grp.itertuples(index=False):
            rec = {"game_id": row.game_id}
            for w in ROLL_WINDOWS:
                prev = hist[-w:]
                if not prev:
                    rec[f"h2h_home_win_rate_{w}"] = np.nan
                    rec[f"h2h_home_goals_for_avg_{w}"] = np.nan
                    rec[f"h2h_home_goals_against_avg_{w}"] = np.nan
                    rec[f"h2h_away_win_rate_{w}"] = np.nan
                    rec[f"h2h_away_goals_for_avg_{w}"] = np.nan
                    rec[f"h2h_away_goals_against_avg_{w}"] = np.nan
                    continue

                home_gf, home_ga, home_win = [], [], []
                for m in prev:
                    if m.home_club_id == row.home_club_id:
                        gf = m.home_club_goals
                        ga = m.away_club_goals
                    else:
                        gf = m.away_club_goals
                        ga = m.home_club_goals
                    home_gf.append(gf)
                    home_ga.append(ga)
                    home_win.append(1.0 if gf > ga else 0.0)

                rec[f"h2h_home_win_rate_{w}"] = float(np.mean(home_win))
                rec[f"h2h_home_goals_for_avg_{w}"] = float(np.mean(home_gf))
                rec[f"h2h_home_goals_against_avg_{w}"] = float(np.mean(home_ga))
                rec[f"h2h_away_win_rate_{w}"] = float(np.mean([1.0 if x < y else 0.0 for x, y in zip(home_gf, home_ga)]))
                rec[f"h2h_away_goals_for_avg_{w}"] = float(np.mean(home_ga))
                rec[f"h2h_away_goals_against_avg_{w}"] = float(np.mean(home_gf))

            rows.append(rec)
            hist.append(row)

    return pd.DataFrame(rows)


def build_team_block(
    games: pd.DataFrame,
    club_match: pd.DataFrame,
    squad_values: pd.DataFrame,
    availability: pd.DataFrame,
    transfers: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    if side not in {"home", "away"}:
        raise ValueError("side must be 'home' or 'away'")

    hosting_value = "Home" if side == "home" else "Away"
    club_col = f"{side}_club_id"

    general = add_rolling_strength_features(club_match, prefix="team_")
    split_base = club_match[club_match["hosting"] == hosting_value].copy()
    split = add_rolling_strength_features(split_base, prefix=f"{side}_split_")

    feats = general[["game_id", "club_id"] + [c for c in general.columns if c.startswith("team_")]].copy()
    feats = feats.merge(
        split[["game_id", "club_id"] + [c for c in split.columns if c.startswith(f"{side}_split_")]],
        on=["game_id", "club_id"],
        how="left",
    )

    feats = feats.merge(availability, on=["game_id", "club_id"], how="left")

    club_date_index = pd.concat(
        [
            games[["game_id", "date", club_col]].rename(columns={club_col: "club_id"}),
        ],
        ignore_index=True,
    )
    feats = feats.merge(club_date_index, on=["game_id", "club_id"], how="left")
    feats = feats.merge(squad_values, on=["club_id", "date"], how="left")
    feats = feats.merge(transfers, on=["club_id", "date"], how="left")

    rename_cols = {
        "club_id": f"{side}_club_id",
        "date": f"{side}_feature_date",
    }
    for c in feats.columns:
        if c in {"game_id", "club_id", "date"}:
            continue
        if c.startswith(f"{side}_split_"):
            rename_cols[c] = c
        else:
            rename_cols[c] = f"{side}_{c}"

    feats = feats.rename(columns=rename_cols)
    return feats


def add_transfer_churn_share(df: pd.DataFrame, side: str) -> pd.DataFrame:
    for wd in TRANSFER_WINDOWS_DAYS:
        churn_col = f"{side}_transfer_squad_churn_count_{wd}d"
        squad_col = f"{side}_clubs_squad_size"
        out_col = f"{side}_transfer_squad_churn_share_{wd}d"
        if churn_col in df.columns and squad_col in df.columns:
            df[out_col] = df[churn_col] / df[squad_col].replace(0, np.nan)
    return df


def build_feature_table(data_dir: Path, max_games: int | None = None) -> pd.DataFrame:
    games = build_games_label_table(data_dir, max_games=max_games)
    club_match = build_club_match_base(games, data_dir)

    availability = build_availability_features(club_match, data_dir)
    squad_values = build_squad_value_features(games, data_dir)
    transfers = build_transfer_features(games, data_dir)
    h2h = build_head_to_head_features(games)

    home_block = build_team_block(games, club_match, squad_values, availability, transfers, side="home")
    away_block = build_team_block(games, club_match, squad_values, availability, transfers, side="away")

    out = games.copy()
    out = out.merge(home_block, on=["game_id", "home_club_id"], how="left")
    out = out.merge(away_block, on=["game_id", "away_club_id"], how="left")
    out = out.merge(h2h, on="game_id", how="left")

    out["home_indicator"] = 1
    out = add_transfer_churn_share(out, side="home")
    out = add_transfer_churn_share(out, side="away")

    out = out.sort_values(["date", "game_id"]).reset_index(drop=True)
    return out


def select_output_columns(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "full":
        return df

    base_cols = [
        "game_id",
        "competition_id",
        "season",
        "round",
        "date",
        "home_club_id",
        "away_club_id",
        "home_club_goals",
        "away_club_goals",
        "home_indicator",
    ]

    keep = list(base_cols)
    for side in ("home", "away"):
        keep.extend(
            [
                f"{side}_team_ppg_3",
                f"{side}_team_goal_diff_avg_3",
                f"{side}_team_goals_scored_avg_3",
                f"{side}_team_goals_conceded_avg_3",
                f"{side}_team_clean_sheet_rate_3",
                f"{side}_team_win_rate_3",
                f"{side}_team_ppg_5",
                f"{side}_team_goal_diff_avg_5",
                f"{side}_team_goals_scored_avg_5",
                f"{side}_team_goals_conceded_avg_5",
                f"{side}_team_clean_sheet_rate_5",
                f"{side}_team_win_rate_5",
                f"{side}_team_ppg_10",
                f"{side}_team_goal_diff_avg_10",
                f"{side}_team_goals_scored_avg_10",
                f"{side}_team_goals_conceded_avg_10",
                f"{side}_team_clean_sheet_rate_10",
                f"{side}_team_win_rate_10",
                f"{side}_team_form_weighted_points_3",
                f"{side}_team_form_weighted_goal_diff_3",
                f"{side}_team_form_weighted_points_5",
                f"{side}_team_form_weighted_goal_diff_5",
                f"{side}_split_ppg_3",
                f"{side}_split_goal_diff_avg_3",
                f"{side}_split_goals_scored_avg_3",
                f"{side}_split_goals_conceded_avg_3",
                f"{side}_split_clean_sheet_rate_3",
                f"{side}_split_win_rate_3",
                f"{side}_split_ppg_5",
                f"{side}_split_goal_diff_avg_5",
                f"{side}_split_goals_scored_avg_5",
                f"{side}_split_goals_conceded_avg_5",
                f"{side}_split_clean_sheet_rate_5",
                f"{side}_split_win_rate_5",
                f"{side}_split_ppg_10",
                f"{side}_split_goal_diff_avg_10",
                f"{side}_split_goals_scored_avg_10",
                f"{side}_split_goals_conceded_avg_10",
                f"{side}_split_clean_sheet_rate_10",
                f"{side}_split_win_rate_10",
                f"{side}_split_form_weighted_points_3",
                f"{side}_split_form_weighted_goal_diff_3",
                f"{side}_split_form_weighted_points_5",
                f"{side}_split_form_weighted_goal_diff_5",
                f"{side}_squad_total_market_value_eur",
                f"{side}_squad_avg_market_value_eur",
                f"{side}_squad_top11_market_value_eur",
                f"{side}_squad_value_per_starter_eur",
                f"{side}_availability_typical_starter_share_5",
                f"{side}_availability_recent_expected_minutes_5",
                f"{side}_availability_heavy_usage_players_5",
                f"{side}_availability_recent_starter_count_5",
                f"{side}_availability_rotation_rate_5",
            ]
        )

        for wd in TRANSFER_WINDOWS_DAYS:
            keep.extend(
                [
                    f"{side}_transfer_in_count_{wd}d",
                    f"{side}_transfer_out_count_{wd}d",
                    f"{side}_transfer_in_fee_{wd}d",
                    f"{side}_transfer_out_fee_{wd}d",
                    f"{side}_transfer_net_spend_{wd}d",
                    f"{side}_transfer_net_incoming_value_{wd}d",
                    f"{side}_transfer_squad_churn_count_{wd}d",
                    f"{side}_transfer_squad_churn_share_{wd}d",
                ]
            )

    for w in ROLL_WINDOWS:
        keep.extend(
            [
                f"h2h_home_win_rate_{w}",
                f"h2h_home_goals_for_avg_{w}",
                f"h2h_home_goals_against_avg_{w}",
                f"h2h_away_win_rate_{w}",
                f"h2h_away_goals_for_avg_{w}",
                f"h2h_away_goals_against_avg_{w}",
            ]
        )

    existing_keep = [c for c in keep if c in df.columns]
    return df[existing_keep].copy()


def main() -> None:
    args = parse_args()
    features = build_feature_table(args.data_dir, max_games=args.max_games)
    features = select_output_columns(features, feature_set=args.feature_set)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)

    print(f"Wrote {len(features):,} rows and {len(features.columns):,} columns to {args.output}")


if __name__ == "__main__":
    main()
