"""
viz_utils.py — Shared display names, colors, and helpers for SERGIO viz scripts.

Import this in any plotting script to get consistent labels and styles:

    from viz_utils import MODEL_LABELS, MODEL_COLORS, RUN_LABELS
"""

# ---------------------------------------------------------------------------
# Run-dir → display name
# Maps state_runs directory names to human-readable labels.
# ---------------------------------------------------------------------------
RUN_LABELS: dict[str, str] = {
    # Pre-pre-training backbones
    "sergio_ppt_v2":          "SERGIO-PPT",
    "sergio_pptcte":          "SERGIO-PPT-CTE",
    "sergio_ppt_reptile":     "SERGIO-REPT",
    "sergio_ppt_reptile_cte": "SERGIO-REPT-CTE",
    "nca_ppt":                "NCA-PPT",

    # Fine-tuned on SERGIO target data
    "spptv2_last_stgt":       "SERGIO-PPT",
    "sergio_tgtv2":           "BASELINE",
    "sergio_tgtv2_cte":       "BASELINE-CTE",
    "nca_stgt_v2":            "NCA",

    # Sample efficiency — spptv2
    "sample_eff_spptv2_cpp10":   "SERGIO-PPT  cpp=10",
    "sample_eff_spptv2_cpp25":   "SERGIO-PPT  cpp=25",
    "sample_eff_spptv2_cpp50":   "SERGIO-PPT  cpp=50",
    "sample_eff_spptv2_cpp100":  "SERGIO-PPT  cpp=100",

    # Sample efficiency — scratch baseline
    "sample_eff_scratch_cpp10":  "BASELINE  cpp=10",
    "sample_eff_scratch_cpp25":  "BASELINE  cpp=25",
    "sample_eff_scratch_cpp50":  "BASELINE  cpp=50",
    "sample_eff_scratch_cpp100": "BASELINE  cpp=100",

    # Sample efficiency — NCA
    "sample_eff_nca_cpp10":      "NCA  cpp=10",
    "sample_eff_nca_cpp25":      "NCA  cpp=25",
    "sample_eff_nca_cpp50":      "NCA  cpp=50",
    "sample_eff_nca_cpp100":     "NCA  cpp=100",

    # Sample efficiency — reptile 1k / 3k / 10k
    "sample_eff_reptile1k_cpp10":   "REPT-1k  cpp=10",
    "sample_eff_reptile1k_cpp25":   "REPT-1k  cpp=25",
    "sample_eff_reptile1k_cpp50":   "REPT-1k  cpp=50",
    "sample_eff_reptile1k_cpp100":  "REPT-1k  cpp=100",
    "sample_eff_reptile1k_full":    "REPT-1k  full",
    "sample_eff_reptile3k_cpp10":   "REPT-3k  cpp=10",
    "sample_eff_reptile3k_cpp25":   "REPT-3k  cpp=25",
    "sample_eff_reptile3k_cpp50":   "REPT-3k  cpp=50",
    "sample_eff_reptile3k_cpp100":  "REPT-3k  cpp=100",
    "sample_eff_reptile3k_full":    "REPT-3k  full",
    "sample_eff_reptile10k_cpp10":  "REPT-10k  cpp=10",
    "sample_eff_reptile10k_cpp25":  "REPT-10k  cpp=25",
    "sample_eff_reptile10k_cpp50":  "REPT-10k  cpp=50",
    "sample_eff_reptile10k_cpp100": "REPT-10k  cpp=100",
    "sample_eff_reptile10k_full":   "REPT-10k  full",

    # Fine-tuned on Replogle-Nadig
    "rpnd_fewshot":              "BASELINE",
    "rpnd_baseline":             "BASELINE",
    "spptv1_rpnd_fewshot":       "SERGIO-PPT-v1",
    "spptv2_rpnd_fewshot":       "SERGIO-PPT",
    "nca_rpnd_fewshot":          "NCA",
    "nca_rpnd":                  "NCA",
    "reptile3k_rpnd_fewshot":    "REPT-3k",
}

# ---------------------------------------------------------------------------
# Condition key (used in plot_sample_efficiency*.py) → display name
# ---------------------------------------------------------------------------
MODEL_LABELS: dict[str, str] = {
    "spptv2":     "SERGIO-PPT",
    "scratch":    "BASELINE",
    "nca":        "NCA",
    "reptile1k":  "REPT-1k",
    "reptile3k":  "REPT-3k",
    "reptile10k": "REPT-10k",
    "scratch@2k": "BASELINE @ 2k steps",
}

# ---------------------------------------------------------------------------
# Condition key → hex color
# ---------------------------------------------------------------------------
MODEL_COLORS: dict[str, str] = {
    "spptv2":     "#2166ac",   # blue
    "scratch":    "#d6604d",   # red-orange
    "nca":        "#4dac26",   # green
    "reptile1k":  "#01bfc4",   # teal
    "reptile3k":  "#8073ac",   # purple
    "reptile10k": "#e08214",   # orange
    "scratch@2k": "#b2182b",   # dark red
}

# ---------------------------------------------------------------------------
# Metric key → display name
# ---------------------------------------------------------------------------
METRIC_LABELS: dict[str, str] = {
    "pearson_delta":          "Pearson Δ (↑)",
    "overlap_at_N":           "Overlap@N (↑)",
    "precision_at_N":         "Precision@N (↑)",
    "discrimination_score_l1":"Discrimination L1 (↑)",
    "mse":                    "MSE (↓)",
    "mae":                    "MAE (↓)",
    "de_nsig_counts_pred":    "DE Sig Counts Pred",
    "de_nsig_counts_real":    "DE Sig Counts Real",
}

# ---------------------------------------------------------------------------
# GRN panel aesthetics (used in plot_grn_*.py)
# ---------------------------------------------------------------------------
GRN_TYPE_ACCENT: dict[str, str] = {
    "BA":    "#f7c948",   # gold
    "BA-VM": "#a78bfa",   # purple
    "ER":    "#38bdf8",   # sky blue
}

GRN_TYPE_FULLNAME: dict[str, str] = {
    "BA":    "Barabási–Albert",
    "BA-VM": "BA Variable-m",
    "ER":    "Erdős–Rényi",
}

# Dark presentation background
BG_COLOR    = "#0f1117"
LABEL_COLOR = "#e8eaf0"
TITLE_COLOR = "#ffffff"
BORDER_COLOR = "#2a2d3a"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Run-dir → condition key (for color lookup)
# ---------------------------------------------------------------------------
_RUN_CONDITION: dict[str, str] = {}
for _run, _label in RUN_LABELS.items():
    for _key, _klabel in MODEL_LABELS.items():
        if _label == _klabel:
            _RUN_CONDITION[_run] = _key
            break


def run_label(run_dir: str) -> str:
    """Return display name for a run dir, falling back to the dir name itself."""
    return RUN_LABELS.get(run_dir, run_dir)


def run_color(run_dir: str, fallback: str = "#888888") -> str:
    """Return the canonical hex color for a run dir."""
    key = _RUN_CONDITION.get(run_dir)
    return MODEL_COLORS.get(key, fallback) if key else fallback
