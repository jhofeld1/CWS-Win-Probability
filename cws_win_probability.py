"""
CWS Finals Win Probability Model — UNC Tar Heels vs Oklahoma Sooners
====================================================================
wOBA = Wieghted On-Base Average. Weights each positive offensive outcome based on linear weights derived from run expectancy per event. 
FIP= Fielding Independent Pitching. Estimates a pitchers ERA in fielding independent outcomes like K,BB, HBP

wOBA weights derived from Fan Graphs


A sabermetric matchup model for the College World Series finals. It derives
team offense from wOBA (linear weights), team pitching from FIP, and converts
the run environment into per-game and best-of-3 series win probabilities.

Outputs:
 CWS_Model_Output.xlsx : hitter/pitcher metrics + win-probability summary
 CWS_Visuals.png       : 4-panel diagnostic + results figure

Run:
  pip install -r requirements.txt
  python cws_win_probability.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "CWS")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HITTERS_CSV = os.path.join(OUTPUT_DIR, "CWS_Hitters.csv")
PITCHERS_CSV = os.path.join(OUTPUT_DIR, "CWS_Pitchers.csv")

# ── OMAHA RUN ENVIRONMENT SCALAR ────────────────────────────
# Calibrates the MLB RE24 baseline to the 2026 CWS Omaha environment.
# Derived from 2026 CWS bracket games runs scored:

#   Combined runs per game (both teams): ~11.6 → 5.8 per team
#   MLB 2023 baseline: 4.61 runs per team per game
#   Raw scalar: 5.8 / 4.61 = 1.26
#   Conservative estimate applied: 1.15
# Tune this value manually — 1.0 = pure MLB baseline
OMAHA_SCALAR = 1.15
# ─────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
# Two distinct weight sets — kept separate so Panel 3 stays a real validity check:
#   BASE_WOBA_WEIGHTS  = wOBA linear-weight coefficients (FanGraphs-style)  → Panel 3 Y-axis
#   BASE_RE24_RUN_VALUES = RE24 state-transition run value per event        → Panel 3 X-axis
# Per the "scale uniformly" decision, OMAHA_SCALAR is applied to BOTH sets, so the
# run environment inflates 15% while the wOBA↔RE24 relationship (and its R²) is preserved.
BASE_WOBA_WEIGHTS = {"BB": 0.69, "HBP": 0.72, "1B": 0.89, "2B": 1.27, "3B": 1.62, "HR": 2.10}
BASE_RE24_RUN_VALUES = {"BB": 0.33, "HBP": 0.35, "1B": 0.47, "2B": 0.77, "3B": 1.07, "HR": 1.40}

# Omaha-adjusted weights actually used by the model
WOBA_WEIGHTS = {k: round(v * OMAHA_SCALAR, 4) for k, v in BASE_WOBA_WEIGHTS.items()}
RE24_RUN_VALUES = {k: round(v * OMAHA_SCALAR, 4) for k, v in BASE_RE24_RUN_VALUES.items()}

# wOBA scale denominator, adjusted for the new run environment
WOBA_SCALE = 1.25 * OMAHA_SCALAR
WOBA_BASELINE = round(0.320 * OMAHA_SCALAR, 4)  # league-avg wOBA, environment-adjusted

FIP_CONSTANT = 3.10          # league-calibration constant for FIP
PYTHAG_EXPONENT = 1.83       # Pythagorean exponent for run-to-win conversion
MIN_IP = 30                  # innings-pitched filter for the pitcher dataset

TEAM_COLORS = {"UNC": "#4B9CD3", "OU": "#841617"}   # Carolina Blue / Crimson
TEAM_LABELS = {"Tar_Heels": "UNC", "Sooners": "OU"}

# Tango MLB run-expectancy matrix (rows = base state, cols = outs 0/1/2)
RE24_MLB_RAW = np.array([
    [0.481, 0.254, 0.098],  # ---
    [0.859, 0.509, 0.224],  # 1--
    [1.100, 0.664, 0.319],  # -2-
    [1.350, 0.950, 0.353],  # --3
    [1.437, 0.884, 0.429],  # 12-
    [1.784, 1.130, 0.478],  # 1-3
    [1.964, 1.376, 0.580],  # -23
    [2.292, 1.541, 0.752],  # 123
])
# Apply the Omaha scalar to every cell so the displayed matrix reflects the
# inflated 2026 CWS run environment rather than the raw MLB baseline.
RE24_MATRIX = np.round(RE24_MLB_RAW * OMAHA_SCALAR, 4)

BASE_STATES = ["---", "1", "2", "3", "1-2", "1-3", "2-3", "123"]
OUT_STATES = ["0 out", "1 out", "2 out"]


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------
def load_data():
    hitters = pd.read_csv(HITTERS_CSV)
    pitchers = pd.read_csv(PITCHERS_CSV)

    # Innings pitched arrive as e.g. 93.2 == 93 + 2/3; convert to true innings.
    def ip_to_float(ip):
        whole = int(ip)
        frac = round((ip - whole) * 10)
        return whole + frac / 3.0

    pitchers["IP_float"] = pitchers["IP"].apply(ip_to_float)
    return hitters, pitchers


def add_woba(hitters):
    """Adds environment-adjusted wOBA using the Omaha-scaled linear weights.

    Because WOBA_WEIGHTS are inflated by OMAHA_SCALAR, the resulting wOBA values
    are environment-adjusted (≈15% above the raw MLB scale), not raw MLB wOBA.
    """
    h = hitters.copy()
    h["1B"] = h["H"] - h["2B"] - h["3B"] - h["HR"]
    numer = (
        WOBA_WEIGHTS["BB"] * h["BB"]
        + WOBA_WEIGHTS["HBP"] * h["HBP"]
        + WOBA_WEIGHTS["1B"] * h["1B"]
        + WOBA_WEIGHTS["2B"] * h["2B"]
        + WOBA_WEIGHTS["3B"] * h["3B"]
        + WOBA_WEIGHTS["HR"] * h["HR"]
    )
    denom = (h["AB"] + h["BB"] + h["HBP"]).replace(0, np.nan)
    h["wOBA"] = (numer / denom).round(3)
    return h


def add_fip(pitchers):
    """FIP using the fields present in the pitcher CSV.

    HR allowed is not provided, so we estimate it from the long-ball rate
    implied by hits and the run environment is captured via the FIP constant.
    """
    p = pitchers.copy()
    # No HR-allowed column; approximate HR allowed from ER/R relationship is noisy,
    # so estimate HR via a modest share of hits (≈9% league HR/H for college).
    est_hr = (p["H"] * 0.09).round(1)
    p["est_HR"] = est_hr
    p["FIP"] = (
        (13 * p["est_HR"] + 3 * (p["BB"] + p["HBP"]) - 2 * p["SO"]) / p["IP_float"]
    ) + FIP_CONSTANT
    p["FIP"] = p["FIP"].round(2)
    return p


# ---------------------------------------------------------------------------
# Win-probability model
# ---------------------------------------------------------------------------
def team_offense_rpg(hitters):
    """Actual runs scored per game for each team, keyed by short team code."""
    rpg = {}
    for raw_team, code in TEAM_LABELS.items():
        sub = hitters[hitters["Team"] == raw_team]
        games = sub["GP"].max()
        runs = sub["R"].sum()
        rpg[code] = runs / games
    return rpg


def team_woba(hitters):
    """PA-weighted environment-adjusted team wOBA, keyed by short team code."""
    out = {}
    for raw_team, code in TEAM_LABELS.items():
        sub = hitters[hitters["Team"] == raw_team]
        out[code] = np.average(sub["wOBA"], weights=sub["PA"])
    return out


def woba_runs_per_game(team_woba_map):
    """wOBA-derived runs/game estimate, environment-adjusted.

    RunsPerGame = (wOBA - WOBA_BASELINE) / WOBA_SCALE * 27, clamped at 0.
    Informational only — the win-probability model below is driven by actual
    runs scored and FIP, so this estimator does not feed the win probabilities.
    """
    return {
        code: max((w - WOBA_BASELINE) / WOBA_SCALE * 27, 0.0)
        for code, w in team_woba_map.items()
    }


def team_pitching_ra9(pitchers):
    """IP-weighted FIP (proxy for runs allowed / 9) for each team's qualified staff."""
    ra9 = {}
    qualified = pitchers[pitchers["IP_float"] >= MIN_IP]
    for raw_team, code in TEAM_LABELS.items():
        sub = qualified[qualified["Team"] == raw_team]
        ra9[code] = np.average(sub["FIP"], weights=sub["IP_float"])
    return ra9


def win_probabilities(off_rpg, def_ra9):
    # Matchup expected runs blend a team's own offense with the opponent's defense.
    unc_runs = np.mean([off_rpg["UNC"], def_ra9["OU"]])
    ou_runs = np.mean([off_rpg["OU"], def_ra9["UNC"]])

    x = PYTHAG_EXPONENT
    p_unc = unc_runs ** x / (unc_runs ** x + ou_runs ** x)
    p_ou = 1 - p_unc

    # Best-of-3 series: P(win) = 3p^2 - 2p^3
    series = lambda p: 3 * p ** 2 - 2 * p ** 3
    return {
        "unc_runs": unc_runs,
        "ou_runs": ou_runs,
        "game_UNC": p_unc,
        "game_OU": p_ou,
        "series_UNC": series(p_unc),
        "series_OU": series(p_ou),
    }


# ---------------------------------------------------------------------------
# Excel output
# ---------------------------------------------------------------------------
def write_excel(hitters, pitchers, probs, off_rpg, def_ra9, woba_rpg, tw):
    path = os.path.join(OUTPUT_DIR, "CWS_Model_Output.xlsx")
    qualified = pitchers[pitchers["IP_float"] >= MIN_IP].copy()

    summary = pd.DataFrame(
        {
            "Metric": [
                "Expected Runs / Game",
                "Offense R/G (actual)",
                "Pitching FIP (IP-weighted, qualified)",
                "Team wOBA (environment-adjusted)",
                "wOBA-derived R/G (informational)",
                "Per-Game Win Probability",
                "Series Win Probability (best-of-3)",
            ],
            "UNC": [
                round(probs["unc_runs"], 3),
                round(off_rpg["UNC"], 3),
                round(def_ra9["UNC"], 3),
                round(tw["UNC"], 3),
                round(woba_rpg["UNC"], 3),
                f"{probs['game_UNC'] * 100:.1f}%",
                f"{probs['series_UNC'] * 100:.1f}%",
            ],
            "OU": [
                round(probs["ou_runs"], 3),
                round(off_rpg["OU"], 3),
                round(def_ra9["OU"], 3),
                round(tw["OU"], 3),
                round(woba_rpg["OU"], 3),
                f"{probs['game_OU'] * 100:.1f}%",
                f"{probs['series_OU'] * 100:.1f}%",
            ],
        }
    )

    # Environment-adjustment sheet so the reader knows the model is NOT using
    # raw MLB weights — it documents the scalar and every adjusted weight.
    env_rows = [
        ["OMAHA_SCALAR", OMAHA_SCALAR, "1.0 = pure MLB baseline"],
        ["wOBA scale (1.25 × scalar)", round(WOBA_SCALE, 4), "denominator for runs/game"],
        ["wOBA baseline (0.320 × scalar)", WOBA_BASELINE, "league-avg wOBA, env-adjusted"],
    ]
    for k in ["BB", "HBP", "1B", "2B", "3B", "HR"]:
        env_rows.append(
            [f"Adj. wOBA weight: {k}", WOBA_WEIGHTS[k], f"raw {BASE_WOBA_WEIGHTS[k]} × {OMAHA_SCALAR}"]
        )
    for k in ["BB", "HBP", "1B", "2B", "3B", "HR"]:
        env_rows.append(
            [f"Adj. RE24 run value: {k}", RE24_RUN_VALUES[k], f"raw {BASE_RE24_RUN_VALUES[k]} × {OMAHA_SCALAR}"]
        )
    env_rows.append(
        ["NOTE", "wOBA values are environment-adjusted", "weights inflated by OMAHA_SCALAR; not raw MLB wOBA"]
    )
    env = pd.DataFrame(env_rows, columns=["Parameter", "Value", "Notes"])

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Win_Probability", index=False)
        env.to_excel(xl, sheet_name="Environment_Adjustment", index=False)
        hitters.sort_values("wOBA", ascending=False).to_excel(
            xl, sheet_name="Hitters", index=False
        )
        qualified.sort_values("FIP").to_excel(xl, sheet_name="Pitchers_Qualified", index=False)
    return path


# ---------------------------------------------------------------------------
# Visualization (4-panel figure)
# ---------------------------------------------------------------------------
def panel_re24(ax):
    sns.heatmap(
        RE24_MATRIX,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=OUT_STATES,
        yticklabels=BASE_STATES,
        cbar_kws={"label": "Run Expectancy"},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title("RE24 Run Expectancy Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Outs")
    ax.set_ylabel("Base State")


def panel_fip_era(ax, pitchers):
    qualified = pitchers[pitchers["IP_float"] >= MIN_IP]
    for raw_team, code in TEAM_LABELS.items():
        sub = qualified[qualified["Team"] == raw_team]
        ax.scatter(sub["ERA"], sub["FIP"], color=TEAM_COLORS[code], label=code, s=70, zorder=3)
        for _, row in sub.iterrows():
            last = row["Player"].split()[-1]
            ax.annotate(last, (row["ERA"], row["FIP"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")

    lo = min(qualified["ERA"].min(), qualified["FIP"].min()) - 0.5
    hi = max(qualified["ERA"].max(), qualified["FIP"].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.6, label="ERA = FIP")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("ERA")
    ax.set_ylabel("FIP")
    ax.set_title("FIP vs ERA by Pitcher (IP ≥ 30)", fontsize=13, fontweight="bold")
    ax.legend(title="Team")


def panel_linear_weights(ax):
    events = list(RE24_RUN_VALUES.keys())
    x = np.array([RE24_RUN_VALUES[e] for e in events])
    y = np.array([WOBA_WEIGHTS[e] for e in events])

    ax.scatter(x, y, color="#333333", s=70, zorder=3)
    for e, xi, yi in zip(events, x, y):
        ax.annotate(e, (xi, yi), fontsize=9, fontweight="bold",
                    xytext=(5, -2), textcoords="offset points")

    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="#841617", lw=2, label="OLS fit")

    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    ax.text(0.05, 0.92, f"$R^2$ = {r2:.4f}", transform=ax.transAxes,
            fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.text(0.05, 0.78,
            "Tight $R^2$ = wOBA coefficients align\nwith run expectancy framework",
            transform=ax.transAxes, fontsize=8, style="italic")
    ax.set_xlabel("RE24 Run Value per Event")
    ax.set_ylabel("wOBA Linear Weight Coefficient")
    ax.set_title(
        "Linear Weights vs RE24 Run Values (wOBA Validity Check)\n"
        f"Weights adjusted for 2026 Omaha run environment (scalar = {OMAHA_SCALAR})",
        fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")


def panel_win_prob(ax, probs):
    groups = ["Per-Game", "Series"]
    unc_vals = [probs["game_UNC"] * 100, probs["series_UNC"] * 100]
    ou_vals = [probs["game_OU"] * 100, probs["series_OU"] * 100]

    xpos = np.arange(len(groups))
    width = 0.35
    bars_unc = ax.bar(xpos - width / 2, unc_vals, width, label="UNC", color=TEAM_COLORS["UNC"])
    bars_ou = ax.bar(xpos + width / 2, ou_vals, width, label="OU", color=TEAM_COLORS["OU"])

    for bars in (bars_unc, bars_ou):
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}%",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(50, color="gray", linestyle="--", alpha=0.7)
    ax.set_xticks(xpos)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Win Probability (%)")
    ax.set_ylim(0, 100)
    ax.set_title("CWS Finals Win Probability — UNC vs Oklahoma",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Team")


def build_figure(pitchers, probs):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    panel_re24(axes[0, 0])
    panel_fip_era(axes[0, 1], pitchers)
    panel_linear_weights(axes[1, 0])
    panel_win_prob(axes[1, 1], probs)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "CWS_Visuals.png")
    fig.savefig(path, dpi=150)
    return fig, path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    hitters, pitchers = load_data()
    hitters = add_woba(hitters)
    pitchers = add_fip(pitchers)

    off_rpg = team_offense_rpg(hitters)
    def_ra9 = team_pitching_ra9(pitchers)
    tw = team_woba(hitters)
    woba_rpg = woba_runs_per_game(tw)
    probs = win_probabilities(off_rpg, def_ra9)

    xlsx_path = write_excel(hitters, pitchers, probs, off_rpg, def_ra9, woba_rpg, tw)
    fig, png_path = build_figure(pitchers, probs)

    print("CWS Finals Win Probability Model")
    print("-" * 52)
    print(f"OMAHA_SCALAR applied : {OMAHA_SCALAR}  (1.0 = pure MLB baseline)")
    print("Adjusted linear weights (env-adjusted, NOT raw MLB):")
    for k in ["BB", "HBP", "1B", "2B", "3B", "HR"]:
        print(f"   {k:<4} wOBA={WOBA_WEIGHTS[k]:<6} | RE24={RE24_RUN_VALUES[k]}")
    print(f"wOBA scale={WOBA_SCALE:.4f}  baseline={WOBA_BASELINE}")
    print("Note: wOBA values below are environment-adjusted.")
    print("-" * 52)
    print(f"Team wOBA (adj) : UNC {tw['UNC']:.3f}  |  OU {tw['OU']:.3f}")
    #print(f"wOBA R/G (info) : UNC {woba_rpg['UNC']:.2f}   |  OU {woba_rpg['OU']:.2f}")
    print(f"Per-game        : UNC {probs['game_UNC']*100:.1f}%  |  OU {probs['game_OU']*100:.1f}%")
    print(f"Series          : UNC {probs['series_UNC']*100:.1f}%  |  OU {probs['series_OU']*100:.1f}%")
    print(f"\nWrote: {xlsx_path}")
    print(f"Wrote: {png_path}")

    # Display inline when run interactively (e.g. in an IPython/Jupyter kernel).
    if hasattr(plt, "isinteractive") and plt.isinteractive():
        plt.show()


if __name__ == "__main__":
    main()
