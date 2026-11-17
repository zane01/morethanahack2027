# Our solution: how to reproduce the submission

This repository is a copy of the official challenge repository with our solution added on top.
Our code is `make_submission.py`; the method is explained in `TECHNICAL_DOCUMENT.md`.

## 1. Environment

```
conda create --name mtah python=3.11
conda activate mtah
pip install -r requirements-solution.txt
```

`requirements-solution.txt` pins the exact library versions we used.

## 2. Data

The match data is gated and is **not** in this repository. Unpack the bundle into `data/`,
keeping its names (see `DATA.md`):

```
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

Only these three files are used. To keep the data elsewhere, set `MTAH_DATA_DIR` (see `games.py`)
or pass `--data-dir`.

## 3. Generate the submission

From the repository root:

```
python make_submission.py
```

This writes `evaluation/results/The Good, the Bad and the Ugly.csv` (199,319 rows,
`frame,label`), named after our team, and prints a short
report (gap categories, label shares, number and length of possessions). Loading the tracking
takes a few minutes; the rest takes seconds.

The pipeline is deterministic (no randomness, no training), so running it again produces the
same file.

## 4. Check it

```
python evaluation/validate_submission.py "evaluation/results/The Good, the Bad and the Ugly.csv"
```

The quotes are needed because the file name contains spaces.
