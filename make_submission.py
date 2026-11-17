#!/usr/bin/env python
"""Frame-by-frame possession labels (H / A / D / X) for BAR-ATH.

    python make_submission.py

Reads the gated match data from data/ (or $MTAH_DATA_DIR) and writes
evaluation/results/The Good, the Bad and the Ugly.csv: 199,319 rows, header "frame,label".
(--team "Another Name" changes the file name.)

The pipeline is fully deterministic -- no randomness, no training, no labels of any kind:

  1. load the tactical Automatic Tracking Data (ATD) and the Smart Tagging
  2. stage 1: every stretch where the ball is not tracked (a "gap") is labelled D (stoppage)
     or X (short gap in open play), from its length, the set pieces and where the ball was
     last seen
  3. per-frame evidence for H / A / D on the frames where the ball is visible: which team is
     closer to the ball, whether the ball is loose, whether play has stopped
  4. a three-state hidden Markov model (H, A, D), decoded with Viterbi, one half at a time
  5. ungraded frames (before kickoff, halftime, after the final whistle) are written as X

Every parameter below is set from football reasoning or from label-free statistics of the
provided data. None was tuned against any ground truth. See TECHNICAL_DOCUMENT.md.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

import games

GAME = "BAR-ATH"
FPS = 25
N_FRAMES = games.GAMES[GAME]["n_video_frames"]          # 199,319
PITCH_L, PITCH_W = games.GAMES[GAME]["pitch"]           # 105.3 x 68.0 m
LABELS = np.array(["H", "A", "D"])
H, A, D = 0, 1, 2

# --- Stage 1: missing-ball gaps ------------------------------------------------------------
LONG_GAP_S = 4.0           # a gap this long is a stoppage (almost no gaps last between 2 and 5 s)
MIN_SETPIECE_GAP_S = 2.0   # a set piece can only be attached to a gap at least this long
MAX_AFTER_S = 10.0         # a set piece may start up to this long after its gap ended
LINE_MARGIN_M = 2.0        # a short gap starting/ending this close to a line = ball out of play

# --- Stage 2: evidence on frames where the ball is visible --------------------------------
K_PER_M = 0.25             # log-odds of H vs A per metre of distance advantage. Low on purpose:
                           # possession must be *controlled*, so an opponent has to stay clearly
                           # closer for about a second before possession changes hands
MAX_ADVANTAGE_M = 5.0      # distance advantages beyond this add nothing
CONTROL_NEAR_M = 1.5       # nearest player this close: full evidence of control
CONTROL_FAR_M = 4.0        # nobody this close: loose ball, no evidence (possession stays put)
SLOW_PLAYERS_MS = 1.5      # mean player speed below this ...
STILL_BALL_MS = 1.0        # ... with a ball slower than this looks like a stoppage
OUT_MARGIN_M = 0.5         # ball this far outside a line is out of play
P_DEAD_OUT = 0.90          # P(D) when the ball is outside the pitch
P_DEAD_STILL = 0.60        # P(D) when the ball is still and the players walk
P_DEAD_PLAY = 0.05         # P(D) otherwise
P_DEAD_PLACED = 0.90       # P(D) between a gap's end and the set piece that restarts play
SETPIECE_BOOST = 3.0       # extra weight for the team a set piece is tagged with ...
SETPIECE_BOOST_S = 2.0     # ... over this long after the restart
P_DEAD_SHORT_GAP = 0.5     # relative weight of D inside a short in-play gap (H and A get 1)

# --- Stage 3: hidden Markov model ----------------------------------------------------------
POSSESSION_MEAN_S = 12.0   # typical length of a possession
SHARE_TO_TURNOVER = 0.5    # when a possession ends: share going to the other team (vs a stoppage)
FLOOR = 1e-9               # smallest emission probability, keeps the logarithms finite


# ==========================================================================================
# Loading
# ==========================================================================================

def numeric(df, columns):
    """Columns as a float array; None and other non-numbers become NaN."""
    return df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def load_tracking(data_dir=None):
    """Ball and player positions in metres, one row per video frame (0 ... 199,318)."""
    from kloppy import metrica

    meta, raw = games.require(["atd_meta", "atd_raw"], GAME, data_dir)
    print("loading the tracking data (this takes a few minutes) ...")
    dataset = metrica.load_tracking_epts(meta_data=meta, raw_data=raw)
    df = dataset.to_df()

    # DATA.md: video_frame = frame_id - 1. Reindexing guarantees exactly one row per video frame.
    df["video_frame"] = pd.to_numeric(df["frame_id"]).astype(int) - 1
    df = df.set_index("video_frame").reindex(np.arange(N_FRAMES))

    teams = {}
    for team in dataset.metadata.teams:
        teams["home" if "home" in str(team.ground).lower() else "away"] = team
    print(f"home (H): {teams['home'].name}   away (A): {teams['away'].name}")

    tracking = {"ball_x": numeric(df, ["ball_x"])[:, 0] * PITCH_L,
                "ball_y": numeric(df, ["ball_y"])[:, 0] * PITCH_W}
    for side in ("home", "away"):
        ids = [p.player_id for p in teams[side].players
               if all(f"{p.player_id}_{c}" in df.columns for c in ("x", "y", "s"))]
        tracking[side + "_x"] = numeric(df, [f"{i}_x" for i in ids]) * PITCH_L
        tracking[side + "_y"] = numeric(df, [f"{i}_y" for i in ids]) * PITCH_W
        tracking[side + "_s"] = numeric(df, [f"{i}_s" for i in ids])        # m/s
    return tracking


def load_set_pieces(data_dir=None):
    """Smart Tagging set pieces: start frame and tagged team (H, A or -1 when N/A)."""
    from kloppy import sportscode

    (path,) = games.require(["smart_tagging"], GAME, data_dir)
    tags = sportscode.load(path).to_df()
    sp = tags[tags["code"].astype(str).str.upper() == "SET PIECES"].copy()
    sp["start_frame"] = (sp["timestamp"].dt.total_seconds() * FPS).round().astype(int)

    team = (sp["Team"] if "Team" in sp.columns else pd.Series("", index=sp.index))
    team = team.astype(str).str.lower()
    # Home is Barcelona (DATA.md). The away team is spelled "Atletic Club" in the tags.
    sp["team"] = np.select([team.str.contains("barcel"), team.str.contains("atl")], [H, A],
                           default=-1)
    return sp.sort_values("start_frame").reset_index(drop=True)


# ==========================================================================================
# Features
# ==========================================================================================

def nanreduce(func, values):
    """np.nanmin / np.nanmean without the warning on rows that are entirely NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(values, axis=1)


def features(tracking):
    bx, by = tracking["ball_x"], tracking["ball_y"]

    def nearest(side):
        d = np.hypot(tracking[side + "_x"] - bx[:, None], tracking[side + "_y"] - by[:, None])
        return nanreduce(np.nanmin, d)

    # The ATD ball_speed column is empty, so the speed is computed from consecutive positions
    # and smoothed with a 5-frame median against tracking jitter.
    step = np.hypot(np.diff(bx, prepend=np.nan), np.diff(by, prepend=np.nan)) * FPS
    ball_speed = pd.Series(step).rolling(5, center=True, min_periods=1).median().to_numpy()

    speeds = np.hstack([tracking["home_s"], tracking["away_s"]])
    return {"ball_x": bx, "ball_y": by,
            "d_home": nearest("home"), "d_away": nearest("away"),
            "ball_speed": ball_speed,
            "mean_speed": nanreduce(np.nanmean, speeds)}


# ==========================================================================================
# Stage 1: label the missing-ball gaps
# ==========================================================================================

def half_index():
    """1 or 2 for graded frames, 0 for pre-match, halftime and after the final whistle."""
    half = np.zeros(N_FRAMES, dtype=int)
    for h, first, last in games.half_video_blocks(GAME):
        half[first:last + 1] = h
    return half


def runs(mask):
    """(starts, ends) of the runs of True in a boolean array; ends are inclusive."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1


def label_gaps(feat, half, sp):
    missing = np.isnan(feat["ball_x"])
    starts, ends = runs(missing & (half > 0))      # halftime separates the halves: no run crosses
    gaps = pd.DataFrame({"start": starts, "end": ends})
    gaps["seconds"] = (gaps["end"] - gaps["start"] + 1) / FPS

    def line_distance(frames):
        """Distance (m) from the ball to the nearest line at these frames; NaN if not seen."""
        x = np.full(len(frames), np.nan)
        y = np.full(len(frames), np.nan)
        ok = (frames >= 0) & (frames < N_FRAMES)
        x[ok], y[ok] = feat["ball_x"][frames[ok]], feat["ball_y"][frames[ok]]
        return np.minimum.reduce([x, PITCH_L - x, y, PITCH_W - y])

    near_line = ((line_distance(gaps["start"].to_numpy() - 1) <= LINE_MARGIN_M) |
                 (line_distance(gaps["end"].to_numpy() + 1) <= LINE_MARGIN_M))

    # Link each set piece to its gap: the gap containing it, or else the one that ended at most
    # MAX_AFTER_S before it (the ball was placed and visible again before the restart).
    gaps["sp_start"] = np.nan
    gaps["sp_team"] = -1
    candidates = gaps[gaps["seconds"] >= MIN_SETPIECE_GAP_S]
    for _, row in sp.iterrows():                    # in time order: the latest set piece wins
        f = row["start_frame"]
        inside = candidates[(candidates["start"] <= f) & (f <= candidates["end"])]
        if len(inside):
            i = inside.index[0]
        else:
            recent = candidates[(candidates["end"] < f) &
                                ((f - candidates["end"]) / FPS <= MAX_AFTER_S)]
            if not len(recent):
                continue
            i = recent["end"].idxmax()
        gaps.loc[i, "sp_start"] = f
        gaps.loc[i, "sp_team"] = row["team"]

    gaps["reason"] = np.select(
        [gaps["seconds"] >= LONG_GAP_S, gaps["sp_start"].notna(), near_line],
        ["long gap", "set piece", "short, near a line"],
        default="short, in play")
    gaps["label"] = np.where(gaps["reason"] == "short, in play", "X", "D")
    return gaps


# ==========================================================================================
# Stage 2: emissions
# ==========================================================================================

def emissions(feat, gaps):
    """N x 3 array: how compatible each frame is with H, A and D."""
    E = np.ones((N_FRAMES, 3))
    visible = ~np.isnan(feat["ball_x"])

    # H vs A: which team has a player closer to the ball, weighted by how close that player is.
    d_home, d_away = feat["d_home"], feat["d_away"]
    nearest = np.fmin(d_home, d_away)
    control = np.clip((CONTROL_FAR_M - nearest) / (CONTROL_FAR_M - CONTROL_NEAR_M), 0, 1)
    control = np.nan_to_num(control)                 # nobody tracked near the ball: no evidence
    advantage = d_away - d_home                       # > 0: a home player is closer
    advantage = np.where(np.isnan(d_home) & ~np.isnan(d_away), -MAX_ADVANTAGE_M, advantage)
    advantage = np.where(np.isnan(d_away) & ~np.isnan(d_home), MAX_ADVANTAGE_M, advantage)
    advantage = np.clip(np.nan_to_num(advantage), -MAX_ADVANTAGE_M, MAX_ADVANTAGE_M)
    p_home = 1 / (1 + np.exp(-K_PER_M * control * advantage))

    # D with a visible ball: ball outside the pitch, or ball still while the players walk and
    # nobody has it at their feet. A slow ball close to a player is a player holding it during a
    # slow build-up (possession), not a stoppage.
    bx, by = feat["ball_x"], feat["ball_y"]
    out = ((bx < -OUT_MARGIN_M) | (bx > PITCH_L + OUT_MARGIN_M) |
           (by < -OUT_MARGIN_M) | (by > PITCH_W + OUT_MARGIN_M))
    still = ((feat["mean_speed"] < SLOW_PLAYERS_MS) & (feat["ball_speed"] < STILL_BALL_MS) &
             (nearest > CONTROL_NEAR_M))
    p_dead = np.select([out, still], [P_DEAD_OUT, P_DEAD_STILL], default=P_DEAD_PLAY)

    visible_E = np.column_stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead])
    E[visible] = visible_E[visible]

    for g in gaps.itertuples():
        if g.label == "X":                            # short gap in play: no evidence
            E[g.start:g.end + 1] = [1.0, 1.0, P_DEAD_SHORT_GAP]
            continue
        E[g.start:g.end + 1] = [FLOOR, FLOOR, 1.0]    # stoppage: D is forced
        if np.isnan(g.sp_start):
            continue
        restart = int(g.sp_start)
        if restart > g.end + 1:                       # ball visible again but still dead
            placed = np.arange(g.end + 1, restart)
            E[placed] = [(1 - P_DEAD_PLACED) / 2, (1 - P_DEAD_PLACED) / 2, P_DEAD_PLACED]
        if g.sp_team in (H, A):                       # the tagged team takes the restart
            start = max(restart, g.end + 1)
            boost = np.arange(start, min(start + int(SETPIECE_BOOST_S * FPS), N_FRAMES))
            E[boost, g.sp_team] *= SETPIECE_BOOST
    return E


# ==========================================================================================
# Stage 3: hidden Markov model
# ==========================================================================================

def transition_matrix(dead_mean_frames):
    """Rows: from H, A, D. A state left with probability p per frame lasts 1/p frames on average."""
    p = 1 / (POSSESSION_MEAN_S * FPS)
    q = 1 / dead_mean_frames
    s = SHARE_TO_TURNOVER
    return np.array([[1 - p, p * s, p * (1 - s)],
                     [p * s, 1 - p, p * (1 - s)],
                     [q / 2, q / 2, 1 - q]])


def viterbi(log_e, log_t):
    """Most likely state sequence, given log-emissions (T x S) and log-transitions (S x S)."""
    n, s = log_e.shape
    back = np.empty((n, s), dtype=np.int8)
    score = np.log(np.full(s, 1 / s)) + log_e[0]
    for t in range(1, n):
        candidates = score[:, None] + log_t             # [from, to]
        back[t] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + log_e[t]
    path = np.empty(n, dtype=int)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


# ==========================================================================================
# Report and main
# ==========================================================================================

def segments(labels):
    """Maximal runs of one label, within each half."""
    rows = []
    for _, first, last in games.half_video_blocks(GAME):
        s = labels[first:last + 1]
        change = np.flatnonzero(s[1:] != s[:-1]) + 1
        for a, b in zip(np.concatenate([[0], change]), np.concatenate([change, [len(s)]])):
            rows.append((s[a], (b - a) / FPS))
    return pd.DataFrame(rows, columns=["label", "seconds"])


def report(labels, gaps, sp):
    print("\nstage 1 -- missing-ball gaps:")
    print(gaps.groupby("reason").agg(gaps=("start", "size"),
                                     frames=("seconds", lambda x: int(round(x.sum() * FPS)))))
    print(f"set pieces linked to a gap: {gaps['sp_start'].notna().sum()} of {len(sp)}")

    graded = labels[half_index() > 0]
    print("\nlabels on graded frames:")
    print(pd.Series(graded).value_counts(normalize=True).mul(100).round(1).to_string())

    seg = segments(labels)
    poss = seg[seg["label"].isin(["H", "A"])]
    print(f"\npossessions: {len(poss)} (H {int((poss['label'] == 'H').sum())}, "
          f"A {int((poss['label'] == 'A').sum())}), "
          f"median {poss['seconds'].median():.1f} s, mean {poss['seconds'].mean():.1f} s")
    print(f"dead-ball stretches: {int((seg['label'] == 'D').sum())}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", default="The Good, the Bad and the Ugly",
                        help="team name; the file is <team>.csv")
    parser.add_argument("--out", default=None, help="output path (default evaluation/results/)")
    parser.add_argument("--data-dir", default=None, help="data folder (default data/)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "evaluation", "results", args.team + ".csv")

    feat = features(load_tracking(args.data_dir))
    sp = load_set_pieces(args.data_dir)
    gaps = label_gaps(feat, half_index(), sp)
    E = emissions(feat, gaps)

    dead = gaps[gaps["label"] == "D"]
    log_t = np.log(transition_matrix((dead["end"] - dead["start"] + 1).mean()))
    log_e = np.log(np.maximum(E, FLOOR))

    labels = np.full(N_FRAMES, "X", dtype="<U1")      # ungraded frames stay X
    for _, first, last in games.half_video_blocks(GAME):
        labels[first:last + 1] = LABELS[viterbi(log_e[first:last + 1], log_t)]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels}).to_csv(out, index=False)

    report(labels, gaps, sp)
    print(f"\nwrote {out} ({N_FRAMES:,} rows)")


if __name__ == "__main__":
    main()
