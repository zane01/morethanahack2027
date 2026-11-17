# Possession from automatic tracking: technical document

**Team:** The Good, the Bad and the Ugly
**Repository:** https://github.com/zane01/morethanahack2027
**Submission file:** `The Good, the Bad and the Ugly.csv`, produced by `python make_submission.py`

## 1. Approach

We label every graded frame H, A or D with a three-state **hidden Markov model** (HMM): the hidden
state is who has the ball (home, away, or nobody because play is stopped), the observations are
per-frame evidence computed from the automatic tracking, and the transitions encode the fact that
possession is sticky. The most likely state sequence is decoded with the **Viterbi algorithm**,
separately for each half.

Why this rather than something else:

- **The ranking metric scores possessions, not frames.** A per-frame classifier, however
  accurate, creates a fake possession every time it flips for a frame. `EVALUATION.md` shows this
  with the baselines: the same nearest-player rule scores spellF1 0.049 frame by frame and 0.243
  once smoothed. Viterbi decodes the most likely *whole sequence*, so isolated noise costs more
  than it is worth while sustained evidence still produces a change, and it places each change
  using evidence from both before and after it.
- **No labels can be used for this game.** The rules forbid hand labels on the provided game and
  training or tuning on the public window. Every parameter is therefore set from football
  reasoning or from label-free statistics of the provided data, and each one is documented in
  the code. The model is small enough that every decision can be explained.
- **Time.** A supervised model trained on other games would need labelled data in a compatible
  format; we chose an interpretable model we could build, check and document within the deadline.

The pipeline has three stages:

1. **Missing-ball gaps**: every stretch where the ball is not tracked gets D or "no evidence".
2. **Evidence on visible frames**: how compatible each frame is with H, A and D.
3. **HMM + Viterbi**: combines the evidence with the transition model.

## 2. Assets used, tried and dropped

| Asset | Use |
|---|---|
| Tactical ATD | **Used.** Ball and player positions, player speeds, team membership of each track slot (from the metadata). |
| Smart Tagging | **Used, set pieces only.** Links stoppages to restarts and names the restarting team. |
| ATD `ball_state`, `ball_owning_team_id` | **Dropped.** Empty in the automatic export. |
| ATD `ball_speed` | **Replaced.** Empty; we compute ball speed from consecutive positions. |
| Linear interpolation of the ball | **Tried and dropped.** Gaps of 2 s or less hold only 0.1% of the missing frames. |
| Other Smart Tagging codes (phases) | **Not used.** Phases are not possessions, and for defending codes we could not verify whether `Team` names the defending or the attacking side. |
| Tactical video, homography | **Used only to look at the data** (projecting tracking onto video, animations), never as input. |
| Broadcast recording | **Not used.** Not aligned to the tactical timeline, a third of its frames have no homography. |

## 3. Training data

**None.** No labelled dataset of any kind was used, from this game or any other. Nothing is
trained.

The public 5-minute ground truth was used only as the rules allow: to check that the pipeline is
correct (frame alignment, file format, that H and A mean what we think). No parameter was chosen
by looking at a score on it.

## 4. Possession rules

### 4.1 Stoppages: what happens when the ball is not tracked

Exploring the tracking gave the central finding of this work: the ball is missing on 40.7% of
graded frames, but in only **150 gaps**, with a median length of **10.7 s** (75% under 21 s,
maximum 221 s). Occlusions in crowds would produce many short gaps; long gaps of this shape look
like stoppages (throw-ins, goal kicks, corners, free kicks, goals, injuries). Two label-free
checks support this:

- the missing share (41%) is close to the share of a typical match in which the ball is out of
  play;
- of the 59 Smart Tagging set pieces, 40 start *inside* a gap and 14 start shortly *after* one
  (median 5.7 s after: the ball is visible again while it is placed or held before the restart).
  Only 5 have no gap nearby. By chance, about 41% would fall inside a gap.

Rules (stage 1), applied to every gap:

| Rule | Label | Gaps | Frames |
|---|---|---|---|
| Gap of at least 4 s | D | 133 | 64,568 |
| Shorter gap linked to a set piece | D | 0 | 0 |
| Short gap that starts or ends within 2 m of a line | D | 2 | 4 |
| Any other short gap (in open play) | no evidence | 15 | 71 |

- **4 s threshold**: almost no gaps last between 2 and 5 s, so the value separates tracking
  flicker from stoppages; a real stoppage rarely lasts under ~4 s.
- **Set-piece link**: a set piece belongs to a gap if it starts inside it, or at most 10 s after
  it ended, and the gap lasts at least 2 s. 47 gaps receive a linked set piece this way.
- **Ball placed before the restart**: when the linked set piece starts after the gap ended, the
  frames in between (ball visible but still dead) are given strong evidence for D.
- **Short in-play gaps** are not written as X: they get neutral evidence and the HMM carries the
  surrounding possession through them. (On graded frames an X can only lose points: spellF1
  replaces it with the previous label anyway, macroF1 counts it as a miss.)

### 4.2 Stoppages with a visible ball

- Ball more than 0.5 m outside the pitch: P(D) = 0.9.
- Ball still (< 1 m/s) while the players walk (mean speed < 1.5 m/s) and nobody is within 1.5 m
  of it: P(D) = 0.6. The last condition matters: a slow ball at a player's feet during a slow
  build-up is possession, not a stoppage. A first version without it produced about 70 short,
  spurious dead-ball stretches that split possessions in two.
- Otherwise P(D) = 0.05.

### 4.3 Control, loose balls and challenges

For each frame with a visible ball we take the distance from the ball to the nearest tracked
player of each team, d_H and d_A, using the team membership of each track slot from the metadata.
Track slots are anonymous and unstable, so only team-level quantities are used.

- **Who is closer**: P(H vs A) = sigmoid(0.25 × w × (d_A − d_H)), with the advantage capped at
  5 m. A 2 m advantage at full weight gives about 62% for the closer team. The weak slope is
  deliberate: a first version with twice the slope produced 585 possessions with a median of
  3.3 s, roughly twice what a match of this length plausibly contains, i.e. it changed hands on
  contacts rather than on controlled possession.
- **Loose balls and balls in flight**: the weight w is 1 when the nearest player is within 1.5 m,
  falls linearly, and is 0 when nobody is within 4 m. A loose ball therefore carries no evidence,
  and the HMM keeps the last team in possession, as the challenge definition requires.
- **Touches and deflections**: a change of possession has to pay the transition cost twice to
  flip and flip back, so a contact of a few frames cannot create a possession; an opponent who
  stays clearly closest to the ball for about a second or more does.

### 4.4 Restarts

When a stoppage is linked to a set piece whose `Team` tag names a team, that team's evidence is
multiplied by 3 for the first 2 s after the restart. Throw-ins are tagged `N/A`, so for them (and
for untagged stoppages) the restarting team comes from the positions alone.

### 4.5 Transitions

A state left with probability p per frame lasts 1/p frames on average, so the transition matrix
is set from typical durations:

- possession: 12 s on average; when it ends, half the time the ball goes to the other team and
  half the time play stops;
- dead ball: the mean length of the stage-1 stoppages (computed from the data, no labels); a
  restart goes to either team with equal probability.

The decoding is not very sensitive to the exact values; what matters is their order of magnitude.

### 4.6 Output

Graded frames get H, A or D from Viterbi. Ungraded frames (before kickoff, halftime, after the
final whistle) are X.

## 5. Diagnostics (label-free)

Printed by `make_submission.py`:

- possessions: 432 (H 216, A 216), median 4.6 s, mean 8.1 s
  (first version: 585 possessions, median 3.3 s, mean 5.9 s)
- share of graded frames: H 33.8 %, A 21.6 %, D 44.6 %
- dead-ball stretches: 194 (first version: 205)

The two parameter changes described in 4.2 and 4.3 were made because of the first version's
possession count and duration, both label-free properties of our own output, and were justified
by the definition of controlled possession. They were not chosen by a score on any ground truth.

The file passes `evaluation/validate_submission.py`. The published floor to beat is the
`baseline_smooth50` spellF1 of 0.2433.

## 6. Limitations and what we would do with more time

- **Nearest player is only a proxy for control.** It does not see body orientation, first touches
  or who actually plays the ball in a crowd.
- **Long live-play gaps are labelled D.** If the ball is lost by the tracker for more than 4 s in
  open play (aerial duels, crowded boxes), we call it a stoppage.
- **Dead ball with a visible ball is weakly detected.** Fouls where the ball stays on the pitch
  rely on the "ball still, players walking" rule, and stoppage boundaries may be off by
  seconds, which spellF1 penalises.
- **Throw-in team** comes only from positions.
- **Parameters are reasoned, not validated.** By design we had no labels to measure them against.

With more time: train the evidence model on other matches with public tracking and event data;
run a ball detector on the video to fill the gaps; use the phase tags as soft possession
evidence once their `Team` semantics are verified; learn the transitions with Baum-Welch; detect
stoppages from player behaviour (collective deceleration, players lining up for a set piece).

## 7. Reproduction

See `SOLUTION.md`. In short:

```
conda create --name mtah python=3.11 && conda activate mtah
pip install -r requirements-solution.txt
# unpack the data bundle into data/
python make_submission.py
python evaluation/validate_submission.py "evaluation/results/The Good, the Bad and the Ugly.csv"
```

The code was written with the help of an AI assistant (Claude, by Anthropic), used for
explanations, code and documentation; the analysis choices and their checks are ours.
