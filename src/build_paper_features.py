#!/usr/bin/env python3
"""
Build a feature table approximating the paper's feature set (Win vs Not Win)

This script uses available repo data to produce `data/paper_match_features.csv`.
It implements:
 - previous-game indicators (W/NW) for home and away
 - previous-game goals for home and away
 - rolling-3 averages for goals and wins (momentum proxies)
 - fatigue proxies: days since previous game and average days between last 3 games
 - optional event-derived stats (shots, shots_on_target, corners, fouls, cards) if `data/game_events.csv` exists and contains them
 - optional weather & kickoff time if `data/weather.csv` provided (otherwise left missing)

The output matches variable names described in the paper where possible.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from feature_extras import build_extra_features


def safe_to_datetime(s):
    return pd.to_datetime(s, errors='coerce')


def load_games(data_dir: Path) -> pd.DataFrame:
    g = pd.read_csv(data_dir / 'games.csv')
    g['date'] = safe_to_datetime(g.get('date'))
    return g


def build_prev_and_rolling(games: pd.DataFrame) -> pd.DataFrame:
    # Use club_games to compute per-club rolling stats
    cg = pd.read_csv(games_path := Path('data') / 'club_games.csv')
    cg = cg.sort_values(['club_id', 'game_id'])
    # attach date
    dates = games[['game_id', 'date']]
    cg = cg.merge(dates, on='game_id', how='left')
    cg['date'] = safe_to_datetime(cg['date'])

    # previous game result (W vs NW) for each club
    cg = cg.sort_values(['club_id', 'date', 'game_id']).reset_index(drop=True)
    cg['prev_win'] = cg.groupby('club_id')['is_win'].shift(1)
    cg['prev_goals'] = cg.groupby('club_id')['own_goals'].shift(1)

    # rolling 3 averages for goals and wins
    cg['avg_goals_3'] = cg.groupby('club_id')['own_goals'].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    cg['wins_last_3'] = cg.groupby('club_id')['is_win'].shift(1).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)

    # days rest features
    cg = cg.sort_values(['club_id', 'date']).reset_index(drop=True)
    cg['prev_date'] = cg.groupby('club_id')['date'].shift(1)
    cg['days_since_prev'] = (cg['date'] - cg['prev_date']).dt.days
    # average days between last 3
    cg['avg_days_last3'] = cg.groupby('club_id')['days_since_prev'].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

    # pivot to home/away per game
    home = cg.rename(columns={'club_id': 'home_club_id',
                              'prev_win': 'home_prev_win',
                              'prev_goals': 'home_prev_goals',
                              'avg_goals_3': 'home_avg_goals_3',
                              'wins_last_3': 'home_wins_last_3',
                              'days_since_prev': 'home_days_since_prev',
                              'avg_days_last3': 'home_avg_days_last3'})
    away = cg.rename(columns={'club_id': 'away_club_id',
                              'prev_win': 'away_prev_win',
                              'prev_goals': 'away_prev_goals',
                              'avg_goals_3': 'away_avg_goals_3',
                              'wins_last_3': 'away_wins_last_3',
                              'days_since_prev': 'away_days_since_prev',
                              'avg_days_last3': 'away_avg_days_last3'})

    # keep relevant cols
    keep_h = ['game_id', 'home_club_id', 'home_prev_win', 'home_prev_goals', 'home_avg_goals_3', 'home_wins_last_3', 'home_days_since_prev', 'home_avg_days_last3']
    keep_a = ['game_id', 'away_club_id', 'away_prev_win', 'away_prev_goals', 'away_avg_goals_3', 'away_wins_last_3', 'away_days_since_prev', 'away_avg_days_last3']

    home_feats = home[keep_h].drop_duplicates(['game_id', 'home_club_id'])
    away_feats = away[keep_a].drop_duplicates(['game_id', 'away_club_id'])

    out = games[['game_id', 'date', 'home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals']].copy()
    out = out.merge(home_feats, on=['game_id', 'home_club_id'], how='left')
    out = out.merge(away_feats, on=['game_id', 'away_club_id'], how='left')

    # previous game W/NW: convert NaN to 0
    for c in ['home_prev_win', 'away_prev_win']:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(int)

    return out


def add_event_stats(out: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    events_path = data_dir / 'game_events.csv'
    if not events_path.exists():
        return out

    # Attempt to compute per-club per-game aggregated stats if columns exist
    # Expecting columns: game_id, club_id, type
    try:
        ev = pd.read_csv(events_path, usecols=['game_id', 'club_id', 'type'])
    except Exception:
        return out

    # Map event types to counters
    mapping = {
        'shot': 'shots',
        'shot_on_target': 'shots_on_target',
        'corner': 'corners',
        'foul': 'fouls',
        'yellow_card': 'yellow_cards',
        'red_card': 'red_cards'
    }

    ev['etype'] = ev['type'].astype(str).str.lower()
    for key in mapping:
        ev[mapping[key]] = (ev['etype'].str.contains(key.replace('_', ' ')) | ev['etype'].str.contains(key)).astype(int)

    agg = ev.groupby(['game_id', 'club_id'])[[v for v in mapping.values()]].sum().reset_index()

    home_agg = agg.rename(columns={'club_id': 'home_club_id',
                                   'shots': 'home_shots', 'shots_on_target': 'home_shots_on_target',
                                   'corners': 'home_corners', 'fouls': 'home_fouls',
                                   'yellow_cards': 'home_yellow_cards', 'red_cards': 'home_red_cards'})
    away_agg = agg.rename(columns={'club_id': 'away_club_id',
                                   'shots': 'away_shots', 'shots_on_target': 'away_shots_on_target',
                                   'corners': 'away_corners', 'fouls': 'away_fouls',
                                   'yellow_cards': 'away_yellow_cards', 'red_cards': 'away_red_cards'})

    out = out.merge(home_agg, on=['game_id', 'home_club_id'], how='left')
    out = out.merge(away_agg, on=['game_id', 'away_club_id'], how='left')

    return out


def add_weather(out: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    wpath = data_dir / 'weather.csv'
    if not wpath.exists():
        # leave weather columns absent
        return out

    w = pd.read_csv(wpath)
    w['date'] = safe_to_datetime(w.get('date'))
    # expected columns: game_id or date+stadium mapping
    if 'game_id' in w.columns:
        out = out.merge(w, on='game_id', how='left')
    else:
        out = out.merge(w, on='date', how='left')
    return out


def merge_extra_features(out: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    extra = build_extra_features(data_dir)
    if extra.empty:
        return out

    keep = [c for c in extra.columns if c != 'game_id']
    extra = extra[['game_id'] + keep].copy()

    # Convert categorical string columns to stable codes so the training script can use them.
    for col in extra.columns:
        if col == 'game_id':
            continue
        if extra[col].dtype == object:
            extra[col] = extra[col].astype('category').cat.codes.replace(-1, np.nan)

    return out.merge(extra, on='game_id', how='left')


def main():
    base = Path(__file__).resolve().parent.parent
    data_dir = base / 'data'
    games = load_games(data_dir)

    out = build_prev_and_rolling(games)
    out = add_event_stats(out, data_dir)
    out = add_weather(out, data_dir)
    out = merge_extra_features(out, data_dir)

    # Target: Win (1) vs Not Win (0)
    out['target'] = (out['home_club_goals'] > out['away_club_goals']).astype(int)

    # Drop rows with missing required rolling features (first few weeks)
    cols_req = ['home_prev_win', 'away_prev_win', 'home_avg_goals_3', 'away_avg_goals_3']
    out = out.dropna(subset=cols_req)

    out_path = data_dir / 'paper_match_features.csv'
    out.to_csv(out_path, index=False)
    print(f'Wrote {len(out):,} rows to {out_path}')


if __name__ == '__main__':
    main()
