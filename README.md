
# CWS Win Probability Model

### 2026 College World Series Finals — UNC Tar Heels vs. Oklahoma Sooners

What I've built is a single-script Python win probability model that is strictly for the 2026 CWS Finals. This model combines weighted on-base average (wOBA), Fielding Independent Pitching (FIP), and a 24-state Run Expectancy matrix that is calibrated to the Omaha run environment for 2026 CWS games. Built as a analytics intern for the Normal CornBelters. Summer 2026.

## Background & Motivation

Standard college baseball win probability tools either borrow raw MLB coefficients without adjustment or rely on aggregate stats that obscure pitching-offense interaction. This model attempts to address both by:


## Context & Motivation

 For starters, this model attempts to combine both offensive and defensive observations to create a win probability model that takes both aspects into consideration. The model does this by:


1. Applying an environment scalar to calibrate the MLB RE baseline to the 2026 Omaha run environment at Charles Schawb field.

2. Using these new Omaha-adjusted RE24 states to build wOBA FanGraphs-style linear weights.

3. Calculating FIP with an estimated HR allowed parameter, BB, HBP, and Ks.

4. Blending each teams scoring and opponents FIP-derived run prevention into a run estimate, that is then used to predict wins per-game and in the series via Pythagorean expectation.
---

## Methodology

### Data

- **CWS_Hitters.csv** — 2026 season hitting stats for qualifying UNC and OU hitters (PA ≥ 150)
- **CWS_Pitchers.csv** — 2026 season pitching stats for qualifying arms (IP ≥ 30)
- Both filtered from full-roster CSVs to remove small samples

### Omaha Run Environment Scalar

The model uses a scalar to adjust the MLB RE24 baseline to the 2026 CWS Omaha environment:

```
2026 CWS bracket results used:
  UNC vs Ole Miss:      6–2
  UNC vs WVU:           5–2
  UNC vs WVU:          12–7
  OU  vs Alabama:       9–0
  OU  vs Georgia:      11–4

Combined runs/game (both teams): ~11.6
Per-team runs/game: 5.80
MLB 2023 baseline:  4.61

Raw scalar:          5.80 / 4.61 = 1.26
Conservative scalar applied: 1.15 (accounts for small sample blowout inflation)
```

`OMAHA_SCALAR = 1.15` is set as a tunable variable at the top of the script.

### RE24 Matrix

A 24-state Base/out RE matrix based of Tom Tango's MLB RE24 Matrix in *The Book*.

### Linear Weights & wOBA

Two weight sets are maintained separately — both scaled by `OMAHA_SCALAR`:

|Event|FanGraphs Base wOBA Weight|Adj wOBA Weight (×1.15)|RE24 Base Run Value|Adj RE24 Run Value (×1.15)|
|:-:|:-:|:-:|:-:|:-:|
|BB|0.69|0.7935|0.33|0.3795|
|HBP|0.72|0.8280|0.35|0.4025|
|1B|0.89|1.0235|0.47|0.5405|
|2B|1.27|1.4605|0.77|0.8855|
|3B|1.62|1.8630|1.07|1.2305|
|HR|2.10|2.4150|1.40|1.6100|

The FanGraphs wOBA weights and RE24 run values are kept as distinct sets so the Panel 3 regression is a validity check (wOBA weights vs. RE24 run values). Both sets are scaled uniformly so the wOBA↔RE24 relationship (and its R²) is preserved under the environment adjustment.

wOBA is calculated at player level using adjusted wOBA weights, then aggregated to team level weighted by PA. Note the denominator used is `AB + BB + HBP` (at-bats plus times on base via walk or HBP), consistent with the events included in the numerator:

```
wOBA = (0.7935×BB + 0.8280×HBP + 1.0235×1B + 1.4605×2B + 1.8630×3B + 2.4150×HR)
       / (AB + BB + HBP)
```

The wOBA-to-runs-per-game estimator uses the environment-adjusted scale and baseline:

```
wOBA_SCALE    = 1.25 × 1.15 = 1.4375
wOBA_BASELINE = 0.320 × 1.15 = 0.368

RunsPerGame = (wOBA - 0.368) / 1.4375 × 27   [clamped at 0]
```

This estimator is **informational only** and does not feed the win probability calculation.

### FIP (Fielding Independent Pitching)

IP-weighted across all qualifying pitchers (IP ≥ 30) per team:

```
FIP = ((13×est_HR + 3×(BB+HBP) - 2×SO) / IP) + 3.10
```

FIP constant of 3.10 used as college baseline. The pitcher CSV does not include HR allowed, so HR is estimated as `H × 0.09` (~9% HR/H, an approximate college average). FIP serves as the team's run-prevention estimate (runs allowed per 9 innings proxy) in the win probability model.

Starter/bullpen splits (GS > 0 = starter) are calculated and exported to Excel for reference but do not separately feed the win probability calculation. 

### Win Probability Model

The model uses a **matchup run estimate** approach rather than a composite scoring system. For each team, expected runs in the matchup are computed by blending that team's actual season runs scored per game with the opponent's IP-weighted FIP:

```python
unc_runs = mean([off_rpg["UNC"], def_ra9["OU"]])
ou_runs  = mean([off_rpg["OU"],  def_ra9["UNC"]])
```

Per-game win probability is then derived via **Pythagorean expectation** with an exponent of 1.83 (Bill James standard):

```
WinProb_UNC = unc_runs^1.83 / (unc_runs^1.83 + ou_runs^1.83)
```

### Series Win Probability (Best of 3)

```
P(win series) = 3p² - 2p³
```

Equivalent to P(win 2–0) + P(win 2–1) = p² + 2p²(1–p).

---

## Results

|Metric|UNC Tar Heels|Oklahoma Sooners|
|:--|:-:|:-:|
|wOBA (adj, env-adjusted)|0.465|0.452|
|wOBA R/G estimate (info only)|1.83|1.58|
|Per-Game Win Probability|**57.2%**|42.8%|
|Series Win Probability|**60.7%**|39.3%|

**Model prediction: UNC Tar Heels win the 2026 CWS Finals.**

Key driver: UNC's pitching staff holds a significant FIP advantage. Glauber, DeCaro, McDuffie, and Rose all sit above the ERA=FIP diagonal in Panel 2, indicating they are outperforming their ERA and likely to sustain or improve in a neutral, high-leverage environment. Almost every OU arm sit below the diagonal with the exception of Johnson (on the diagonal), suggesting ERA may be flattering their underlying indicators going into a short series.

---

## Visualizations (`CWS_Visuals.png`)

Four-panel matplotlib/seaborn figure (16×12 in, 150 DPI):

| Panel                         | Description                                                                                                                                                                                                                                                                                                                                                                         |
| :---------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RE24 Heatmap**              | 24-state run expectancy matrix scaled to Omaha environment (OMAHA_SCALAR = 1.15). Color: blue (low) → red (high). Bases loaded, 0 out = 2.64 expected runs.                                                                                                                                                                                                                         |
| **FIP vs ERA Scatter**        | Per-pitcher scatter (IP ≥ 30) with ERA=FIP reference line. Above diagonal = FIP > ERA (pitcher outperforming ERA). UNC = Carolina Blue (#4B9CD3), OU = Crimson (#841617).                                                                                                                                                                                                           |
| **Linear Weights Regression** | Adjusted wOBA coefficients (Y) plotted against adjusted RE24 run values (X) per event. OLS fit shown with R² = 0.9989. Note: high R² is structurally expected — FanGraphs wOBA weights are themselves derived from RE24 frameworks. The regression is a validity check for internal consistency, not an independent validation. Subtitle confirms that theOmaha scalar was applied. |
| **Win Probability Dashboard** | Per-game and series win probability grouped bar chart for both teams. 50% reference line shown. Values annotated on bars.                                                                                                                                                                                                                                                           |

---

## Limitations & Known Assumptions
- **SEC vs. ACC Run environment and statistics.** The model compares performance and player stats from two different conferences in college baseball with different competition levels. While both conferences are extremely strong, they differ in terms of the depth of pitching staffs, which can alter a player's statistical output and overall performance.
- **RE24 matrix is MLB-derived.** The Omaha scalar (1.15) partially corrects for the college run environment but is estimated from only 5 bracket games — a small sample inflated by two blowouts (12–7, 11–4).
- **HR allowed is estimated.** The pitcher CSV has no HR-allowed column so the model approximates it as `H × 0.09`. This introduces noise into FIP for individual pitchers but is partially mitigated by IP-weighting across a full staff.
- **wOBA denominator is AB + BB + HBP, not PA.** Sacrifice flies and sacrifice hits are excluded from both numerator and denominator, which is standard but worth noting.
- **wOBA R/G is informational only.** Win probability is driven by actual runs scored per game and IP-weighted FIP — not the wOBA-derived run estimator.
- **No park factor beyond scalar.** Charles Schwab Field wind direction and speed can materially affect HR rates but is not modeled explicitly.
- **No lineup order effects.** Offense is treated as a pool; run expectancy is not simulated through a specific batting order.
- **No fatigue or bullpen availability.** Days of rest and recent bullpen usage are not incorporated. This is notable since both staffs pitched deep into the bracket before the Finals.
- **Small qualifier sample for OU.** PA ≥ 150 and IP ≥ 30 thresholds exclude OU's hottest recent hitters (Brumbaugh .385, Davis .375) who have only 15 and 10 plate appearances. Still, their short-sample numbers would likely push OU's offensive score higher.

---

## Excel Output (`CWS_Model_Output.xlsx`)

|Sheet|Contents|
|:--|:--|
|`Win_Probability`|Per-game and series win probabilities, expected runs, wOBA, FIP per team|
|`Environment_Adjustment`|OMAHA_SCALAR value, all adjusted wOBA weights and RE24 run values with notes|
|`Hitters`|Full hitter table sorted by wOBA descending|
|`Pitchers_Qualified`|Qualified pitchers (IP ≥ 30) sorted by FIP ascending|

--- 


To adjust the run environment scalar:

```python
OMAHA_SCALAR = 1.15  # 1.0 = pure MLB baseline, 1.26 = raw 2026 CWS estimate
```

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
openpyxl
```

---

## Author

Built by Jonah Hofeld — Baseball Analytics Intern, Normal CornBelters (Prospect League, 2026). Applied Mathematics & Data Science, Boston University (Expected May 2028).

---


## How to run

```bash
pip install -r requirements.txt
python cws_win_probability.py
```

