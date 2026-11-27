"""Fig. 2: benchmark of CRESCENT against eight comparators across common and rare conditions.

Redraws manuscript Fig. 2 from the aggregate tables in `source_data/` and writes
TIFF, PDF and SVG to `output/`. No UK Biobank access is needed.

    python figures/figure2/plot.py

The code comes from the notebook `Figure1_model_comparison_publication_layout`
(the notebook calls this "Figure 1"; it is Fig. 2 in the manuscript). Cells are copied
unchanged, with cell numbers kept, apart from the input and output paths in cells 1, 2
and 20. Internal names that start `FIG1_` were kept for fidelity.

`source_data/` is filled on the HPC by `collect_source_data.py`.
"""

import matplotlib

matplotlib.use("Agg")  # draw without a display; files are written to output/


# =============================================================================
# Notebook cell 1
# =============================================================================

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Repository paths. The notebook searched for the HPC project root; here every
# input is read from source_data/ next to this script and every output goes to output/.
HERE = Path(__file__).resolve().parent
SOURCE_DATA = Path(os.environ.get("FIG2_SOURCE_DATA", HERE / "source_data")).expanduser().resolve()
OUTPUT_DIR = HERE / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT = SOURCE_DATA          # con_abb.json is read from source_data/Other_data/
os.chdir(OUTPUT_DIR)                # audit CSVs written to the working directory land in output/

# CRESCENT run used for this figure (published fit, CRESCENT_Final).
CRESCENT_RUN_DIR = SOURCE_DATA / "CRESCENT"

CRESCENT_METRICS_PATH = (
    CRESCENT_RUN_DIR / "test_per_condition_metrics_by_horizon.csv"
)

# The frozen overall-prevalence table is read from source_data/.
COHORT_RUN_DIR = SOURCE_DATA
CINDEX_TAU_MONTHS = 60

print("Source data:", SOURCE_DATA)
print("CRESCENT metrics:", CRESCENT_METRICS_PATH)
print("Output folder:", OUTPUT_DIR)


# =============================================================================
# Notebook cell 2
# =============================================================================

# Edit this ordered mapping to add or remove comparators.
# Repository version: each model's locked-test result tables are read from
# source_data/<model>/. collect_source_data.py records the run each one came from.
MODELS = {
    model: SOURCE_DATA / model / "test_per_condition_metrics_by_horizon.csv"
    for model in (
        "Cox",
        "XGBoostSurvival",
        "Survtrace",
        "DDH",
        "Motor",
        "Delphi-2M",
        "Gemma",
        "CRESCENT-PT",
    )
}
MODELS["CRESCENT"] = CRESCENT_METRICS_PATH

MODEL_INTEGRATED_FILES = {
    model: Path(path).parent / "test_integrated_metrics_by_condition.csv"
    for model, path in MODELS.items()
}
MODEL_CINDEX_FILES = {
    model: Path(path).parent / "test_cindex_by_condition_with_ci.csv"
    for model, path in MODELS.items()
}
MODEL_MACRO_CI_FILES = {
    model: Path(path).parent / "test_macro_metrics_with_ci.csv"
    for model, path in MODELS.items()
}

# Publication mode refuses a model that lacks any required *core* table or
# panel. C-index is configured separately below as an optional panel.
FIG1_REQUIRE_COMPLETE_MODELS = True
FIG1_REQUIRE_IDENTICAL_TEST_COUNTS = True
# What to do when a compared model has no locked-test row for a condition that
# CRESCENT has: "drop" removes that condition for ALL models (so every model is
# compared on the same condition set) and records it; "raise" stops the run.
# Differing event COUNTS for the same condition always stop the run.
FIG1_MISSING_ROW_POLICY = "drop"
FIG1_CI_LEVEL = 0.95
PRIMARY_RISK_TYPE_FOR_REVISED = "death_adjusted"


# Conditions with fewer than this number of locked-test events at the
# reference horizon are excluded before common/rare row selection.
REF_HORIZON = 60
TEST_COUNT_HORIZONS_MONTHS = (60,)
FIG1_MIN_TEST_EVENTS = 10

# The displayed rows are fixed before looking at model performance.
TOP_N_COMMON = 10
BOTTOM_N_RARE = 10

# HIV-related targets are excluded from every Figure 1 condition population:
# displayed rows, win counts, and aggregate point/CI summaries.
FIG1_EXCLUDED_CONDITION_PATTERNS = (
    r"\bHIV\b",
    r"human immunodeficiency",
)
FIG1_EXCLUSION_REASON = "prespecified HIV-related target exclusion"

# Overall prevalence can be supplied as a frozen CSV. If it is absent, this
# notebook derives it from the locked train/validation/test canonical samples:
# a participant is counted for condition k when it was recorded by the index
# (`at_risk[k] == 0`) or is recorded later (`isfinite(t_tilde[k])`).
OVERALL_PREVALENCE_PATH = (
    COHORT_RUN_DIR / "figure1_overall_condition_prevalence.csv"
)
CONDITION_VOCAB_PATH = (
    PROJECT_ROOT / "cell3_condition_vocab_nodeath.csv"
)
CANONICAL_SPLIT_SAMPLE_FILES = {
    "train": COHORT_RUN_DIR / "canonical_train_samples.pkl",
    "validation": COHORT_RUN_DIR / "canonical_val_samples.pkl",
    "test": COHORT_RUN_DIR / "canonical_test_samples.pkl",
}
# Integration window for Figure 2 panels D-E, in months after the six-month entry.
# (0, 60) covers recruitment months 6-66. It must match integration_start_months /
# integration_end_months in the macro-CI and integrated-metric tables.
ENTRY_DELAY_MONTHS = 6
INTEGRATION_WINDOW_AFTER_ENTRY = (0, 60)
INTEGRATION_WINDOW_RECRUITMENT = tuple(m + ENTRY_DELAY_MONTHS for m in INTEGRATION_WINDOW_AFTER_ENTRY)


# Insertion order is the Figure 1 panel order. `direction` is +1 when larger
# values are better and -1 when smaller values are better.
FIG1_PANEL_SPECS = {
    "2-year AUROC": {
        "source": "per_condition",
        "horizon_months": 24,
        "column": "auroc",
        "macro_metric": "auc_ipcw_cr",
        "macro_horizon_months": 24,
        "direction": 1,
        "format": ".2f",
        "reference_line": 0.5,
        "bounds": (0.0, 1.0),
    },
    "5-year AUROC": {
        "source": "per_condition",
        "horizon_months": 60,
        "column": "auroc",
        "macro_metric": "auc_ipcw_cr",
        "macro_horizon_months": 60,
        "direction": 1,
        "format": ".2f",
        "reference_line": 0.5,
        "bounds": (0.0, 1.0),
    },
    f"C-index (τ={CINDEX_TAU_MONTHS} mo)": {
    "source": "cindex",
    "column": "cindex_ipcw_cr",
    "horizon_months": CINDEX_TAU_MONTHS,
    "macro_metric": "cindex_ipcw_cr",
    "macro_horizon_months": CINDEX_TAU_MONTHS,
    "direction": 1,
    "format": ".2f",
    "reference_line": 0.5,
    "bounds": (0.0, 1.0),
},
    "Scaled integrated Brier": {
        "source": "integrated",
        "column": "scaled_ibs_ipcw_ht",
        "macro_metric": "scaled_ibs_ipcw_ht",
        "macro_integration_months": INTEGRATION_WINDOW_AFTER_ENTRY,
        "direction": 1,
        "format": ".2f",
        "reference_line": 0.0,
        "bounds": (None, 1.0),
    },
    "Integrated calibration error": {
        "source": "integrated",
        "column": "integrated_abs_calibration_error",
        "macro_metric": "integrated_abs_calibration_error",
        "macro_integration_months": INTEGRATION_WINDOW_AFTER_ENTRY,
        "direction": -1,
        "format": ".3f",
        "reference_line": 0.0,
        "bounds": (0.0, None),
    },
}

FIG1_PANEL_DISPLAY_LABELS = {
    "2-year AUROC": "2-year AUROC ↑",
    "5-year AUROC": "5-year AUROC ↑",
    f"C-index (τ={CINDEX_TAU_MONTHS} mo)": (
        f"C-index, 5 years ↑"
    ),
    "Scaled integrated Brier": "Scaled IBS skill score ↑",
    "Integrated calibration error": "Integrated calibration error ↓",
}

# C-index is currently available for only a subset of models. Keeping this
# policy explicit allows those models to remain in every other panel while
# preserving NaN placeholders in the C-index heatmap and macro footer.
FIG1_OPTIONAL_PANELS = frozenset(
    panel_name
    for panel_name, specification in FIG1_PANEL_SPECS.items()
    if specification["source"] == "cindex"
)
FIG1_REFERENCE_MODEL = "CRESCENT"


# =============================================================================
# Notebook cell 5
# =============================================================================

def first_present(columns, candidates, label, model_name):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"{model_name}: cannot resolve {label}; expected one of "
        f"{list(candidates)}, found {list(columns)}"
    )


def normalise_per_condition_table(
    frame,
    model_name,
    risk_type=PRIMARY_RISK_TYPE_FOR_REVISED,
):
    """Return the legacy-compatible `df_all` schema required by Figure 2."""
    frame = frame.copy()
    columns = set(frame.columns)

    # Historical table format.
    old_required = {
        "section",
        "condition",
        "horizon_months",
        "auroc",
        "n_pos",
    }
    if old_required.issubset(columns) and "condition_name" not in columns:
        keep = [
            "section",
            "condition",
            "horizon_months",
            "auroc",
            "n_pos",
        ]
        if "cindex" in frame.columns:
            keep.append("cindex")
        if "known_n" in frame.columns:
            keep.append("known_n")
        elif "n" in frame.columns:
            keep.append("n")
        out = frame.loc[:, keep].copy()
        out["Model"] = model_name
        out["section"] = (
            out["section"].astype(str).str.strip().str.lower()
        )
        out["condition"] = out["condition"].astype(str).str.strip()
        for column in ("horizon_months", "auroc", "n_pos"):
            out[column] = pd.to_numeric(out[column], errors="coerce")
        if "cindex" in out.columns:
            out["cindex"] = pd.to_numeric(
                out["cindex"], errors="coerce"
            )
        return out

    required = {
        "condition_name",
        "horizon_months",
        "risk_type",
        "event_n",
    }
    missing = required.difference(columns)
    if missing:
        raise ValueError(
            f"{model_name}: per-condition table missing "
            f"{sorted(missing)}"
        )

    frame = frame[
        frame["risk_type"].astype(str).str.strip().eq(str(risk_type))
    ].copy()
    if frame.empty:
        raise ValueError(
            f"{model_name}: no per-condition rows for risk_type={risk_type!r}"
        )

    auc_column = first_present(
        frame.columns,
        ("auc_ipcw_cr", "auroc"),
        "competing-risk AUROC",
        model_name,
    )
    cindex_column = next(
        (
            column
            for column in ("cindex_ipcw_cr", "ctd_ipcw", "cindex")
            if column in frame.columns
        ),
        None,
    )
    known_column = next(
        (
            column
            for column in ("eligible_n", "known_n", "n")
            if column in frame.columns
        ),
        None,
    )

    within = pd.DataFrame({
        "section": "within",
        "condition": (
            frame["condition_name"].astype(str).str.strip()
        ),
        "horizon_months": pd.to_numeric(
            frame["horizon_months"], errors="coerce"
        ),
        "auroc": pd.to_numeric(
            frame[auc_column], errors="coerce"
        ),
        "n_pos": pd.to_numeric(
            frame["event_n"], errors="coerce"
        ),
        "Model": model_name,
    })
    if known_column is not None:
        within["n"] = pd.to_numeric(
            frame[known_column], errors="coerce"
        )

    parts = [within]
    if cindex_column is not None:
        cindex = pd.DataFrame({
            "condition": (
                frame["condition_name"].astype(str).str.strip()
            ),
            "ctd": pd.to_numeric(
                frame[cindex_column], errors="coerce"
            ),
            "n_pos": pd.to_numeric(
                frame["event_n"], errors="coerce"
            ),
        })
        cindex = (
            cindex.dropna(subset=["condition", "ctd"])
            .groupby("condition", as_index=False)
            .agg(
                auroc=("ctd", "mean"),
                n_pos=("n_pos", "max"),
            )
        )
        cindex["section"] = "cindex"
        cindex["horizon_months"] = -1
        cindex["Model"] = model_name
        parts.append(cindex[[
            "section",
            "condition",
            "horizon_months",
            "auroc",
            "n_pos",
            "Model",
        ]])

    out = pd.concat(parts, ignore_index=True, sort=False)
    out["section"] = (
        out["section"].astype(str).str.strip().str.lower()
    )
    out["condition"] = out["condition"].astype(str).str.strip()
    return out


def normalise_integrated_table(
    frame,
    model_name,
    risk_type=PRIMARY_RISK_TYPE_FOR_REVISED,
):
    frame = frame.copy()
    required = {
        "condition_name",
        "risk_type",
        "scaled_ibs_ipcw_ht",
        "integrated_abs_calibration_error",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{model_name}: integrated table missing {sorted(missing)}"
        )
    frame = frame[
        frame["risk_type"].astype(str).str.strip().eq(str(risk_type))
    ].copy()
    if frame.empty:
        raise ValueError(
            f"{model_name}: no integrated rows for risk_type={risk_type!r}"
        )
    out = pd.DataFrame({
        "condition": (
            frame["condition_name"].astype(str).str.strip()
        ),
        "Model": model_name,
        "scaled_ibs_ipcw_ht": pd.to_numeric(
            frame["scaled_ibs_ipcw_ht"], errors="coerce"
        ),
        "integrated_abs_calibration_error": pd.to_numeric(
            frame["integrated_abs_calibration_error"],
            errors="coerce",
        ),
    })
    return (
        out.groupby(["condition", "Model"], as_index=False)
        .mean(numeric_only=True)
    )


def normalise_cindex_table(
    frame,
    model_name,
    risk_type=PRIMARY_RISK_TYPE_FOR_REVISED,
    tau_months=CINDEX_TAU_MONTHS,
):
    """Load the exact prespecified truncated competing-risk C-index."""
    frame = frame.copy()
    required = {
        "condition_name",
        "metric",
        "estimate",
        "horizon_months",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{model_name}: C-index table missing {sorted(missing)}"
        )
    if "risk_type" in frame.columns:
        frame = frame[
            frame["risk_type"].astype(str).str.strip().eq(str(risk_type))
        ].copy()
    frame["horizon_months"] = pd.to_numeric(
        frame["horizon_months"], errors="coerce"
    )
    frame = frame[
        frame["metric"].astype(str).str.strip().eq("cindex_ipcw_cr")
        & frame["horizon_months"].eq(int(tau_months))
    ].copy()
    if frame.empty:
        raise ValueError(
            f"{model_name}: no cindex_ipcw_cr rows at tau={tau_months} months"
        )
    out = pd.DataFrame({
        "condition": frame["condition_name"].astype(str).str.strip(),
        "Model": model_name,
        "cindex_ipcw_cr": pd.to_numeric(
            frame["estimate"], errors="coerce"
        ),
    })
    if not np.isfinite(out["cindex_ipcw_cr"].to_numpy(float)).any():
        raise ValueError(
           f"{model_name}: no finite {tau_months}-month C-index estimates"
        )
    duplicated = out.duplicated(["condition", "Model"], keep=False)
    if duplicated.any():
        duplicate_conditions = sorted(
            out.loc[duplicated, "condition"].unique().tolist()
        )
        raise ValueError(
            f"{model_name}: duplicate {tau_months}-month C-index rows for "
            f"{duplicate_conditions[:10]}"
        )
    return out


def normalise_macro_ci_table(
    frame,
    model_name,
    panel_specs=FIG1_PANEL_SPECS,
    risk_type=PRIMARY_RISK_TYPE_FOR_REVISED,
    optional_panels=FIG1_OPTIONAL_PANELS,
):
    """Map macro estimates/CIs, retaining NaNs for optional missing panels."""
    frame = frame.copy()
    optional_panels = set(optional_panels)
    required = {
        "metric",
        "estimate",
        "ci_lower",
        "ci_upper",
        "ci_level",
        "ci_method",
        "ci_status",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{model_name}: macro-CI table missing {sorted(missing)}"
        )
    if "risk_type" in frame.columns:
        frame = frame[
            frame["risk_type"].astype(str).str.strip().eq(str(risk_type))
        ].copy()

    for column in (
        "horizon_months",
        "integration_start_months",
        "integration_end_months",
        "estimate",
        "ci_lower",
        "ci_upper",
        "fixed_condition_n",
    ):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    rows = []
    for panel_name, specification in panel_specs.items():
        selected = frame[
            frame["metric"].astype(str).str.strip().eq(
                specification["macro_metric"]
            )
        ].copy()
        if "macro_horizon_months" in specification:
            selected = selected[
                selected["horizon_months"].eq(
                    int(specification["macro_horizon_months"])
                )
            ]
        else:
            start, end = specification["macro_integration_months"]
            selected = selected[
                selected["integration_start_months"].eq(int(start))
                & selected["integration_end_months"].eq(int(end))
            ]
        if len(selected) == 0 and panel_name in optional_panels:
            rows.append({
                "panel": panel_name,
                "Model": model_name,
                "estimate": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "fixed_condition_n": np.nan,
                "ci_level": FIG1_CI_LEVEL,
                "ci_method": "",
                "availability": "missing",
            })
            continue
        if len(selected) != 1:
            raise ValueError(
                f"{model_name}: expected one macro-CI row for "
                f"{panel_name!r}, found {len(selected)}"
            )
        record = selected.iloc[0]
        if str(record["ci_status"]).strip().lower() != "ok":
            if panel_name in optional_panels:
                rows.append({
                    "panel": panel_name,
                    "Model": model_name,
                    "estimate": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "fixed_condition_n": np.nan,
                    "ci_level": record.get("ci_level", FIG1_CI_LEVEL),
                    "ci_method": record.get("ci_method", ""),
                    "availability": "unavailable",
                })
                continue
            raise ValueError(
                f"{model_name}: macro CI is unavailable for {panel_name!r}: "
                f"ci_status={record['ci_status']!r}"
            )
        values = pd.to_numeric(
            record[["estimate", "ci_lower", "ci_upper"]],
            errors="coerce",
        )
        if not np.isfinite(values.to_numpy(float)).all():
            if panel_name in optional_panels:
                rows.append({
                    "panel": panel_name,
                    "Model": model_name,
                    "estimate": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "fixed_condition_n": np.nan,
                    "ci_level": record.get("ci_level", FIG1_CI_LEVEL),
                    "ci_method": record.get("ci_method", ""),
                    "availability": "unavailable",
                })
                continue
            raise ValueError(
                f"{model_name}: non-finite macro estimate/CI for "
                f"{panel_name!r}"
            )
        if not np.isclose(
            float(record["ci_level"]),
            float(FIG1_CI_LEVEL),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{model_name}: {panel_name!r} CI level is "
                f"{record['ci_level']!r}, expected {FIG1_CI_LEVEL}."
            )
        if "participant_bootstrap" not in str(record["ci_method"]):
            raise ValueError(
                f"{model_name}: {panel_name!r} does not use the required "
                f"participant bootstrap: {record['ci_method']!r}"
            )
        if float(values["ci_lower"]) > float(values["ci_upper"]):
            raise ValueError(
                f"{model_name}: macro CI bounds are reversed for "
                f"{panel_name!r}."
            )
        rows.append({
            "panel": panel_name,
            "Model": model_name,
            "estimate": float(values["estimate"]),
            "ci_lower": float(values["ci_lower"]),
            "ci_upper": float(values["ci_upper"]),
            "fixed_condition_n": record["fixed_condition_n"],
            "ci_level": record.get("ci_level", np.nan),
            "ci_method": record.get("ci_method", ""),
            "availability": "available",
        })
    return pd.DataFrame(rows)


# =============================================================================
# Notebook cell 7
# =============================================================================

per_condition_parts = []
per_condition_loaded = []
load_failures = {}

for model_name, path in MODELS.items():
    path = Path(path)
    if not path.is_file():
        load_failures[model_name] = (
            f"missing per-condition table: {path}"
        )
        continue
    try:
        normalised = normalise_per_condition_table(
            pd.read_csv(path),
            model_name,
        )
    except Exception as error:
        load_failures[model_name] = (
            f"per-condition table {path}: "
            f"{type(error).__name__}: {error}"
        )
        continue
    per_condition_parts.append(normalised)
    per_condition_loaded.append(model_name)

if not per_condition_parts:
    raise RuntimeError(
        "No model per-condition performance tables were loaded."
    )

df_all = pd.concat(
    per_condition_parts,
    ignore_index=True,
    sort=False,
)
df_all["condition"] = df_all["condition"].astype(str).str.strip()
df_all["Model"] = df_all["Model"].astype(str).str.strip()
df_all["section"] = (
    df_all["section"].astype(str).str.strip().str.lower()
)
for column in ("horizon_months", "auroc", "n_pos"):
    df_all[column] = pd.to_numeric(
        df_all[column], errors="coerce"
    )


# -----------------------------------------------------------------------------
# Load the integrated tables from the matching model run directories.
# -----------------------------------------------------------------------------

integrated_parts = []
integrated_loaded = []

for model_name in per_condition_loaded:
    path = MODEL_INTEGRATED_FILES[model_name]
    if not path.is_file():
        load_failures[model_name] = (
            f"missing integrated table: {path}"
        )
        continue
    try:
        normalised = normalise_integrated_table(
            pd.read_csv(path),
            model_name,
        )
    except Exception as error:
        load_failures[model_name] = (
            f"integrated table {path}: "
            f"{type(error).__name__}: {error}"
        )
        continue
    integrated_parts.append(normalised)
    integrated_loaded.append(model_name)

# -----------------------------------------------------------------------------
# Load the optional exact 60-month C-index and required macro CI tables.
# -----------------------------------------------------------------------------

cindex_parts = []
cindex_loaded = []
cindex_unavailable = {}
macro_ci_parts = []
macro_ci_loaded = []
macro_cindex_loaded = []

for model_name in per_condition_loaded:
    cindex_path = MODEL_CINDEX_FILES[model_name]
    if not cindex_path.is_file():
        cindex_unavailable[model_name] = (
            f"missing C-index table: {cindex_path}"
        )
    else:
        try:
            normalised_cindex = normalise_cindex_table(
    pd.read_csv(cindex_path),
    model_name,
    tau_months=CINDEX_TAU_MONTHS,
)
        except Exception as error:
            cindex_unavailable[model_name] = (
                f"C-index table {cindex_path}: "
                f"{type(error).__name__}: {error}"
            )
        else:
            cindex_parts.append(normalised_cindex)
            cindex_loaded.append(model_name)

    macro_ci_path = MODEL_MACRO_CI_FILES[model_name]
    if not macro_ci_path.is_file():
        previous = load_failures.get(model_name, "")
        load_failures[model_name] = (
            previous + (" | " if previous else "")
            + f"missing macro-CI table: {macro_ci_path}"
        )
    else:
        try:
            normalised_macro_ci = normalise_macro_ci_table(
                pd.read_csv(macro_ci_path),
                model_name,
            )
        except Exception as error:
            previous = load_failures.get(model_name, "")
            load_failures[model_name] = (
                previous + (" | " if previous else "")
                + f"macro-CI table {macro_ci_path}: "
                f"{type(error).__name__}: {error}"
            )
        else:
            macro_ci_parts.append(normalised_macro_ci)
            macro_ci_loaded.append(model_name)
            optional_macro = normalised_macro_ci[
                normalised_macro_ci["panel"].isin(FIG1_OPTIONAL_PANELS)
            ]
            if (
                len(optional_macro)
                and optional_macro[
                    ["estimate", "ci_lower", "ci_upper"]
                ].notna().to_numpy().all()
            ):
                macro_cindex_loaded.append(model_name)

complete_models = [
    model
    for model in MODELS
    if (
        model in per_condition_loaded
        and model in integrated_loaded
        and model in macro_ci_loaded
    )
]
incomplete_models = [
    model for model in MODELS if model not in complete_models
]

print("FIGURE 1 MODEL-OUTPUT LOAD REPORT")
print("-" * 88)
print(f"Core-complete models: {len(complete_models)}/{len(MODELS)}")
for model in complete_models:
    cindex_point_status = (
        "available" if model in cindex_loaded else "blank"
    )
    cindex_macro_status = (
        "available" if model in macro_cindex_loaded else "blank"
    )
    print(
        f"  ✓ {model}: "
        "AUROC + integrated metrics + required macro CIs; "
        f"C-index point={cindex_point_status}, "
        f"macro CI={cindex_macro_status}"
    )
for model in incomplete_models:
    print(f"  ✗ {model}: {load_failures.get(model, 'incomplete')}")
for model in complete_models:
    if model in cindex_unavailable:
        print(
            f"  · {model}: C-index will be blank "
            f"({cindex_unavailable[model]})"
        )
print("-" * 88)

if FIG1_REQUIRE_COMPLETE_MODELS and incomplete_models:
    details = "\n".join(
        f"  - {model}: {load_failures.get(model, 'incomplete')}"
        for model in incomplete_models
    )
    raise RuntimeError(
        "Locked Figure 1 requires per-condition AUROC, integrated metrics, "
        "and participant-bootstrap macro CIs for every non-optional panel. "
        "C-index is optional and is not the cause of this failure. Fix the "
        "required inputs reported below:\n" + details
    )
if not complete_models:
    raise RuntimeError(
        "No model has all required core Figure 1 tables."
    )

loaded_models = complete_models
df_integrated_all = pd.concat(
    integrated_parts,
    ignore_index=True,
    sort=False,
)
df_integrated_all = df_integrated_all[
    df_integrated_all["Model"].isin(loaded_models)
].copy()
df_cindex_all = (
    pd.concat(
        cindex_parts,
        ignore_index=True,
        sort=False,
    )
    if cindex_parts
    else pd.DataFrame(
        columns=["condition", "Model", "cindex_ipcw_cr"]
    )
)
df_cindex_all = df_cindex_all[
    df_cindex_all["Model"].isin(loaded_models)
].copy()
macro_ci_all = pd.concat(
    macro_ci_parts,
    ignore_index=True,
    sort=False,
)
macro_ci_all = macro_ci_all[
    macro_ci_all["Model"].isin(loaded_models)
].copy()

if FIG1_REFERENCE_MODEL not in loaded_models:
    raise RuntimeError(
        f"{FIG1_REFERENCE_MODEL} must have all required core Figure 1 tables."
    )


# -----------------------------------------------------------------------------
# Build the separate five-metric Figure 1 long table.
# -----------------------------------------------------------------------------

metric_parts = []

for panel_name, specification in FIG1_PANEL_SPECS.items():
    if specification["source"] == "per_condition":
        source = df_all[
            df_all["section"].eq("within")
            & df_all["Model"].isin(loaded_models)
            & df_all["horizon_months"].eq(
                specification["horizon_months"]
            )
        ][["condition", "Model", "auroc", "n_pos"]].copy()
        source = source.rename(columns={"auroc": "value"})
    elif specification["source"] == "cindex":
        source = df_cindex_all[
            ["condition", "Model", specification["column"]]
        ].copy()
        source = source.rename(
            columns={specification["column"]: "value"}
        )
        source["n_pos"] = np.nan
    elif specification["source"] == "integrated":
        source = df_integrated_all[
            ["condition", "Model", specification["column"]]
        ].copy()
        source = source.rename(
            columns={specification["column"]: "value"}
        )
        source["n_pos"] = np.nan
    else:
        raise ValueError(
            f"Unknown Figure 1 source {specification['source']!r}"
        )
    source["panel"] = panel_name
    source["direction"] = int(specification["direction"])
    metric_parts.append(source)

fig1_metric_long = pd.concat(
    metric_parts,
    ignore_index=True,
    sort=False,
)
fig1_metric_long["value"] = pd.to_numeric(
    fig1_metric_long["value"], errors="coerce"
)
fig1_metric_long = (
    fig1_metric_long
    .groupby(
        ["panel", "condition", "Model"],
        as_index=False,
        sort=False,
    )
    .agg(
        value=("value", "mean"),
        n_pos=("n_pos", "max"),
    )
)

# Materialise every panel/condition/model combination. Missing C-index inputs
# therefore remain explicit NaNs in exports as well as blank cells in the plot.
fig1_condition_universe = sorted(
    df_all.loc[
        df_all["section"].eq("within")
        & df_all["Model"].eq(FIG1_REFERENCE_MODEL),
        "condition",
    ].dropna().astype(str).unique().tolist()
)
fig1_full_index = pd.MultiIndex.from_product(
    [
        list(FIG1_PANEL_SPECS),
        fig1_condition_universe,
        loaded_models,
    ],
    names=["panel", "condition", "Model"],
)
fig1_metric_long = (
    fig1_metric_long
    .set_index(["panel", "condition", "Model"])
    .reindex(fig1_full_index)
    .reset_index()
)
fig1_metric_long["direction"] = (
    fig1_metric_long["panel"]
    .map({
        panel_name: int(specification["direction"])
        for panel_name, specification in FIG1_PANEL_SPECS.items()
    })
    .astype(int)
)

coverage = (
    fig1_metric_long.dropna(subset=["value"])
    .groupby(["panel", "Model"])["condition"]
    .nunique()
    .unstack(fill_value=0)
    .reindex(
        index=list(FIG1_PANEL_SPECS),
        columns=loaded_models,
        fill_value=0,
    )
)
required_coverage_failures = [
    f"{panel}/{model}"
    for panel in coverage.index
    if panel not in FIG1_OPTIONAL_PANELS
    for model in coverage.columns
    if int(coverage.loc[panel, model]) == 0
]
if required_coverage_failures:
    raise RuntimeError(
        "Figure 1 has no finite values for required panel/model pairs: "
        + ", ".join(required_coverage_failures)
    )
optional_coverage_gaps = [
    f"{panel}/{model}"
    for panel in coverage.index
    if panel in FIG1_OPTIONAL_PANELS
    for model in coverage.columns
    if int(coverage.loc[panel, model]) == 0
]
if optional_coverage_gaps:
    print(
        "Optional Figure 1 values will be blank: "
        + ", ".join(optional_coverage_gaps)
    )


# =============================================================================
# Notebook cell 10
# =============================================================================

def _normalise_overall_prevalence(frame):
    frame = frame.copy()
    condition_column = first_present(
        frame.columns,
        ("condition", "condition_name", "condition_code"),
        "overall-prevalence condition name",
        "overall prevalence",
    )
    count_column = next(
        (
            column
            for column in (
                "overall_diagnosed_n",
                "overall_prevalence_n",
                "prevalent_n",
                "count",
            )
            if column in frame.columns
        ),
        None,
    )
    denominator_column = next(
        (
            column
            for column in (
                "overall_cohort_n",
                "cohort_n",
                "denominator_n",
                "n_total",
            )
            if column in frame.columns
        ),
        None,
    )
    prevalence_column = next(
        (
            column
            for column in (
                "overall_prevalence",
                "prevalence",
                "prevalence_fraction",
            )
            if column in frame.columns
        ),
        None,
    )
    if prevalence_column is None and (
        count_column is None or denominator_column is None
    ):
        raise ValueError(
            "Overall-prevalence CSV needs either a prevalence column or both "
            "a count and denominator column."
        )

    out = pd.DataFrame({
        "condition": frame[condition_column].astype(str).str.strip(),
        "overall_diagnosed_n": (
            pd.to_numeric(frame[count_column], errors="coerce")
            if count_column is not None
            else np.nan
        ),
        "overall_cohort_n": (
            pd.to_numeric(frame[denominator_column], errors="coerce")
            if denominator_column is not None
            else np.nan
        ),
        "overall_prevalence": (
            pd.to_numeric(frame[prevalence_column], errors="coerce")
            if prevalence_column is not None
            else np.nan
        ),
    })
    if prevalence_column is None:
        out["overall_prevalence"] = (
            out["overall_diagnosed_n"] / out["overall_cohort_n"]
        )
    if out["condition"].duplicated().any():
        duplicates = sorted(
            out.loc[out["condition"].duplicated(False), "condition"].unique()
        )
        raise ValueError(
            "Overall-prevalence table has duplicate conditions: "
            f"{duplicates[:10]}"
        )
    if (
        out["condition"].eq("").any()
        or out["overall_prevalence"].isna().any()
        or (~out["overall_prevalence"].between(0.0, 1.0)).any()
    ):
        raise ValueError(
            "Overall-prevalence table contains blank condition names or "
            "invalid prevalence values."
        )
    return out


def _derive_overall_prevalence_from_locked_cohort():
    vocab_path = Path(CONDITION_VOCAB_PATH).expanduser()
    # The CRESCENT run saves the exact condition list (and order) its targets
    # were built with. Prefer it: the project-level vocabulary CSV can describe
    # a different condition set and then no longer matches the sample width.
    metadata_path = Path(COHORT_RUN_DIR) / "canonical_target_metadata.json"
    use_run_metadata = metadata_path.is_file()
    if not use_run_metadata and not vocab_path.is_file():
        raise FileNotFoundError(
            "Cannot derive overall prevalence: neither "
            f"{metadata_path} nor the condition vocab {vocab_path} exists."
        )
    missing_samples = [
        str(path)
        for path in CANONICAL_SPLIT_SAMPLE_FILES.values()
        if not Path(path).is_file()
    ]
    if missing_samples:
        raise FileNotFoundError(
            "Cannot derive overall prevalence because canonical split files "
            "are missing:\n" + "\n".join(missing_samples)
        )

    if use_run_metadata:
        _meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        condition_names_overall = [
            str(x).strip()
            for x in (_meta["condition_names"] if isinstance(_meta, dict) else _meta)
        ]
        if len(set(condition_names_overall)) != len(condition_names_overall):
            raise ValueError(f"Duplicate condition names in {metadata_path}.")
        print(f"Condition list ({len(condition_names_overall)}) from {metadata_path}")
    else:
        vocab = pd.read_csv(vocab_path)
        name_column = first_present(
            vocab.columns,
            ("condition_name", "condition_code", "condition"),
            "condition name",
            "condition vocabulary",
        )
        if "condition_id" in vocab.columns:
            vocab["condition_id"] = pd.to_numeric(
                vocab["condition_id"], errors="raise"
            ).astype(int)
            vocab = vocab.sort_values("condition_id")
            expected_ids = np.arange(len(vocab), dtype=int)
            if not np.array_equal(
                vocab["condition_id"].to_numpy(int), expected_ids
            ):
                raise ValueError(
                    "Condition vocabulary IDs must be consecutive from zero."
                )
        condition_names_overall = (
            vocab[name_column].astype(str).str.strip().tolist()
        )
    condition_n = len(condition_names_overall)
    diagnosed_counts = np.zeros(condition_n, dtype=np.int64)
    cohort_n = 0

    for split_name, path in CANONICAL_SPLIT_SAMPLE_FILES.items():
        samples = pd.read_pickle(path)
        required_columns = {"eid", "at_risk", "t_tilde"}
        missing = required_columns.difference(samples.columns)
        if missing:
            raise ValueError(
                f"{split_name} canonical samples missing {sorted(missing)}"
            )
        if samples["eid"].astype(str).duplicated().any():
            raise ValueError(
                f"{split_name} canonical samples are not one row per participant."
            )
        at_risk = np.stack(samples["at_risk"].to_numpy()).astype(bool)
        t_tilde = np.stack(samples["t_tilde"].to_numpy()).astype(float)
        if at_risk.shape != t_tilde.shape or at_risk.shape[1] != condition_n:
            raise ValueError(
                f"{split_name} canonical target width {at_risk.shape[1]} does not "
                f"match the condition list ({condition_n}) from "
                f"{metadata_path if use_run_metadata else vocab_path}."
            )
        ever_recorded = (~at_risk) | np.isfinite(t_tilde)
        diagnosed_counts += ever_recorded.sum(axis=0).astype(np.int64)
        cohort_n += int(len(samples))

    derived = pd.DataFrame({
        "condition": condition_names_overall,
        "overall_diagnosed_n": diagnosed_counts,
        "overall_cohort_n": int(cohort_n),
        "overall_prevalence": diagnosed_counts / float(cohort_n),
        "selection_source": (
            "locked train+validation+test cohort; recorded before index or "
            "during observed follow-up"
        ),
    })
    OVERALL_PREVALENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(OVERALL_PREVALENCE_PATH, index=False)
    print(
        "Derived and saved frozen overall-prevalence table:",
        OVERALL_PREVALENCE_PATH,
    )
    return derived


# =============================================================================
# Notebook cell 11
# =============================================================================

if OVERALL_PREVALENCE_PATH.is_file():
    overall_prevalence = _normalise_overall_prevalence(
        pd.read_csv(OVERALL_PREVALENCE_PATH)
    )
    overall_prevalence_source = str(OVERALL_PREVALENCE_PATH)
else:
    overall_prevalence = _normalise_overall_prevalence(
        _derive_overall_prevalence_from_locked_cohort()
    )
    overall_prevalence_source = (
        str(OVERALL_PREVALENCE_PATH) + " (derived in this run)"
    )

metric_condition_universe = set(
    fig1_metric_long.loc[
        fig1_metric_long["Model"].eq("CRESCENT"),
        "condition",
    ].astype(str)
)
missing_from_metrics = sorted(
    set(overall_prevalence["condition"]) - metric_condition_universe
)
missing_from_prevalence = sorted(
    metric_condition_universe - set(overall_prevalence["condition"])
)
if missing_from_metrics or missing_from_prevalence:
    raise ValueError(
        "The overall-prevalence and CRESCENT metric condition universes must "
        "match exactly. "
        f"Only in prevalence={missing_from_metrics[:10]}; "
        f"only in metrics={missing_from_prevalence[:10]}"
    )
overall_prevalence = overall_prevalence[
    overall_prevalence["condition"].isin(metric_condition_universe)
].copy()

overall_prevalence = overall_prevalence.sort_values(
    ["overall_prevalence", "overall_diagnosed_n", "condition"],
    ascending=[False, False, True],
    na_position="last",
).reset_index(drop=True)
overall_prevalence["overall_prevalence_rank"] = (
    np.arange(1, len(overall_prevalence) + 1)
)

# Locked-test event counts are resolved before row selection so the minimum
# support rule applies to both the common and rare prevalence groups.
reference_count_rows = df_all[
    df_all["section"].eq("within")
    & df_all["Model"].eq("CRESCENT")
    & df_all["horizon_months"].isin(TEST_COUNT_HORIZONS_MONTHS)
][["condition", "horizon_months", "n_pos"]].copy()
reference_count_rows = (
    reference_count_rows
    .groupby(["condition", "horizon_months"], as_index=False)["n_pos"]
    .max()
)
expected_reference_pairs = pd.MultiIndex.from_product(
    [
        sorted(metric_condition_universe),
        list(TEST_COUNT_HORIZONS_MONTHS),
    ],
    names=["condition", "horizon_months"],
)
observed_reference_pairs = pd.MultiIndex.from_frame(
    reference_count_rows[["condition", "horizon_months"]]
)
missing_reference_pairs = expected_reference_pairs.difference(
    observed_reference_pairs
)
if len(missing_reference_pairs):
    raise RuntimeError(
        "CRESCENT is missing locked-test count rows required for Figure 1: "
        f"{list(missing_reference_pairs[:10])}"
    )
npos_by_h = reference_count_rows.pivot(
    index="condition",
    columns="horizon_months",
    values="n_pos",
).reindex(columns=TEST_COUNT_HORIZONS_MONTHS)
ref_counts = pd.to_numeric(
    npos_by_h[REF_HORIZON], errors="coerce"
).astype("Int64")
if ref_counts.isna().any():
    missing_count_conditions = ref_counts[ref_counts.isna()].index.tolist()
    raise RuntimeError(
        "Locked-test reference event counts are missing for: "
        + ", ".join(missing_count_conditions[:10])
    )

# -----------------------------------------------------------------------------
# Conditions for which any compared model has no locked-test row.
# -----------------------------------------------------------------------------
_ref_pairs = set(zip(
    reference_count_rows["condition"], reference_count_rows["horizon_months"]
))
fig1_missing_row_records = []
for _model in loaded_models:
    if _model == "CRESCENT":
        continue
    _cand = df_all[
        df_all["section"].eq("within")
        & df_all["Model"].eq(_model)
        & df_all["horizon_months"].isin(TEST_COUNT_HORIZONS_MONTHS)
    ]
    _cand_pairs = set(zip(_cand["condition"], _cand["horizon_months"]))
    for _cond, _h in sorted(_ref_pairs - _cand_pairs):
        fig1_missing_row_records.append(
            {"model": _model, "condition": _cond, "horizon_months": _h}
        )
fig1_missing_row_table = pd.DataFrame(
    fig1_missing_row_records, columns=["model", "condition", "horizon_months"]
)
fig1_missing_row_conditions = sorted(
    fig1_missing_row_table["condition"].unique().tolist()
)
if fig1_missing_row_conditions:
    print(
        f"{len(fig1_missing_row_conditions)} condition(s) have no locked-test "
        "row for at least one compared model:"
    )
    print(fig1_missing_row_table.to_string(index=False))
    if FIG1_MISSING_ROW_POLICY == "raise":
        raise RuntimeError(
            "Missing per-condition rows (set FIG1_MISSING_ROW_POLICY='drop' "
            "to exclude these conditions for every model)."
        )
    fig1_missing_row_table.to_csv(
        "Figure1_conditions_dropped_missing_model_rows.csv", index=False
    )
    print("These conditions are excluded for ALL models; saved "
          "Figure1_conditions_dropped_missing_model_rows.csv")
else:
    fig1_missing_row_table.to_csv(
        "Figure1_conditions_dropped_missing_model_rows.csv", index=False
    )
_missing_set = set(fig1_missing_row_conditions)

fig1_exclusion_regex = "|".join(FIG1_EXCLUDED_CONDITION_PATTERNS)
overall_prevalence["figure1_pattern_excluded"] = (
    overall_prevalence["condition"]
    .astype(str)
    .str.contains(
        fig1_exclusion_regex,
        case=False,
        regex=True,
        na=False,
    )
)
overall_prevalence["test_event_n_reference_horizon"] = (
    overall_prevalence["condition"].map(ref_counts.to_dict()).astype("Int64")
)
overall_prevalence["figure1_below_min_test_events"] = (
    overall_prevalence["test_event_n_reference_horizon"]
    .lt(int(FIG1_MIN_TEST_EVENTS))
    .fillna(True)
)
overall_prevalence["figure1_excluded"] = (
    overall_prevalence["figure1_pattern_excluded"]
    | overall_prevalence["figure1_below_min_test_events"]
)
overall_prevalence["figure1_exclusion_reason"] = ""
overall_prevalence.loc[
    overall_prevalence["figure1_pattern_excluded"],
    "figure1_exclusion_reason",
] = FIG1_EXCLUSION_REASON
low_support_reason = (
    f"fewer than {FIG1_MIN_TEST_EVENTS} locked-test events at "
    f"{REF_HORIZON} months"
)
low_support_only = (
    overall_prevalence["figure1_below_min_test_events"]
    & ~overall_prevalence["figure1_pattern_excluded"]
)
overall_prevalence.loc[
    low_support_only,
    "figure1_exclusion_reason",
] = low_support_reason
both_exclusions = (
    overall_prevalence["figure1_below_min_test_events"]
    & overall_prevalence["figure1_pattern_excluded"]
)
overall_prevalence.loc[
    both_exclusions,
    "figure1_exclusion_reason",
] = (
    FIG1_EXCLUSION_REASON + "; " + low_support_reason
)
_miss_mask = overall_prevalence["condition"].isin(_missing_set)
overall_prevalence["figure1_missing_model_rows"] = _miss_mask
overall_prevalence.loc[_miss_mask, "figure1_excluded"] = True
overall_prevalence.loc[_miss_mask, "figure1_exclusion_reason"] = (
    overall_prevalence.loc[_miss_mask, "figure1_exclusion_reason"]
    .astype(str)
    .map(lambda r: (r + "; " if r.strip() else "")
         + "no locked-test row for at least one compared model")
)
fig1_excluded_conditions = sorted(
    overall_prevalence.loc[
        overall_prevalence["figure1_excluded"],
        "condition",
    ].astype(str).tolist()
)
fig1_pattern_excluded_conditions = sorted(
    overall_prevalence.loc[
        overall_prevalence["figure1_pattern_excluded"],
        "condition",
    ].astype(str).tolist()
)
fig1_low_support_conditions = sorted(
    overall_prevalence.loc[
        overall_prevalence["figure1_below_min_test_events"],
        "condition",
    ].astype(str).tolist()
)
selection_prevalence = overall_prevalence.loc[
    ~overall_prevalence["figure1_excluded"]
].copy()
selection_prevalence["figure1_selection_rank"] = (
    np.arange(1, len(selection_prevalence) + 1)
)
overall_prevalence["figure1_selection_rank"] = (
    overall_prevalence["condition"].map(
        selection_prevalence.set_index("condition")[
            "figure1_selection_rank"
        ]
    )
)

if len(selection_prevalence) < TOP_N_COMMON + BOTTOM_N_RARE:
    raise RuntimeError(
        "Not enough non-excluded conditions for non-overlapping top and "
        f"bottom groups: {len(selection_prevalence)} available."
    )

if fig1_excluded_conditions:
    print(
        "Figure 1 display exclusions applied before prevalence ranking: "
        f"{len(fig1_excluded_conditions)} condition(s). Examples:",
        fig1_excluded_conditions[:20],
    )

common_conds = (
    selection_prevalence.head(TOP_N_COMMON)["condition"].tolist()
)
rare_conds = (
    selection_prevalence.tail(BOTTOM_N_RARE)["condition"].tolist()
)
if set(common_conds).intersection(rare_conds):
    raise RuntimeError("Top and bottom prevalence groups overlap.")

# Displayed condition order for the five panels.
top_conds = common_conds + rare_conds
displayed_below_threshold = [
    condition
    for condition in top_conds
    if int(ref_counts.loc[condition]) < int(FIG1_MIN_TEST_EVENTS)
]
if displayed_below_threshold:
    raise RuntimeError(
        "Figure 1 selected conditions below the locked-test support threshold: "
        + ", ".join(displayed_below_threshold)
    )
condition_group = {
    **{condition: "10 most prevalent" for condition in common_conds},
    **{condition: "10 least prevalent" for condition in rare_conds},
}
overall_prevalence_map = overall_prevalence.set_index(
    "condition"
)["overall_prevalence"].to_dict()
overall_prevalence_n_map = overall_prevalence.set_index(
    "condition"
)["overall_diagnosed_n"].to_dict()

# Confirm that every comparator reports the same locked-test event counts as
# the CRESCENT reference table used for selection and annotations.
fig1_extra_row_records = []
count_mismatches = []
for model_name in loaded_models:
    if model_name == "CRESCENT":
        continue
    candidate = (
        df_all[
            df_all["section"].eq("within")
            & df_all["Model"].eq(model_name)
            & df_all["horizon_months"].isin(TEST_COUNT_HORIZONS_MONTHS)
        ]
        .groupby(["condition", "horizon_months"])["n_pos"]
        .max()
    )
    reference = reference_count_rows.set_index(
        ["condition", "horizon_months"]
    )["n_pos"]
    reference = reference[
        ~reference.index.get_level_values("condition").isin(_missing_set)
    ]
    candidate = candidate[
        ~candidate.index.get_level_values("condition").isin(_missing_set)
    ]
    # Rows a comparator has for conditions that CRESCENT does not model are not
    # part of the common condition set: ignore them (and record them) when the
    # policy is "drop"; otherwise they count as a mismatch.
    _extra_all = candidate.index.difference(reference.index)
    if FIG1_MISSING_ROW_POLICY == "drop" and len(_extra_all):
        fig1_extra_row_records.extend(
            {"model": model_name, "condition": _c, "horizon_months": _h}
            for _c, _h in _extra_all
        )
        candidate = candidate.loc[~candidate.index.isin(_extra_all)]
    missing_pairs = reference.index.difference(candidate.index)
    extra_pairs = candidate.index.difference(reference.index)
    if len(missing_pairs) or len(extra_pairs):
        count_mismatches.append(
            f"{model_name}: missing {len(missing_pairs)} and extra "
            f"{len(extra_pairs)} condition/horizon count rows"
        )
    aligned = pd.concat(
        [reference.rename("reference"), candidate.rename("candidate")],
        axis=1,
        join="inner",
    ).dropna()
    different = aligned[
        ~np.isclose(
            aligned["reference"].to_numpy(float),
            aligned["candidate"].to_numpy(float),
            rtol=0.0,
            atol=0.0,
        )
    ]
    if len(different):
        count_mismatches.append(
            f"{model_name}: {len(different)} condition/horizon pairs"
        )
fig1_extra_row_table = pd.DataFrame(
    fig1_extra_row_records, columns=["model", "condition", "horizon_months"]
)
fig1_extra_row_conditions = sorted(
    fig1_extra_row_table["condition"].unique().tolist()
)
fig1_extra_row_table.to_csv(
    "Figure1_comparator_conditions_not_in_CRESCENT_ignored.csv", index=False
)
if fig1_extra_row_conditions:
    print(
        f"{len(fig1_extra_row_conditions)} condition(s) reported by comparators "
        "but not modelled by CRESCENT are ignored for the figure: "
        f"{fig1_extra_row_conditions}"
    )
if count_mismatches:
    message = (
        "Locked test event counts differ across models. Counts cannot be "
        "treated as one common test-set annotation:\n"
        + "\n".join(count_mismatches)
    )
    if FIG1_REQUIRE_IDENTICAL_TEST_COUNTS:
        raise RuntimeError(message)
    print("WARNING:", message)

ref_counts = npos_by_h[REF_HORIZON].astype("Int64")
npos_map = ref_counts.to_dict()


# =============================================================================
# Notebook cell 13
# =============================================================================

panel_labels = list(FIG1_PANEL_SPECS)
panel_directions = {
    panel: int(spec["direction"]) for panel, spec in FIG1_PANEL_SPECS.items()
}
m_order = list(loaded_models)
REFERENCE_MODEL = FIG1_REFERENCE_MODEL
all_conds = common_conds + rare_conds

full_panel_data = {}
panel_data = {}
for panel in panel_labels:
    rows = fig1_metric_long.loc[fig1_metric_long["panel"].eq(panel)]
    matrix = (
        rows.pivot_table(
            index="condition", columns="Model", values="value", aggfunc="mean"
        )
        .reindex(columns=m_order)
    )
    full_panel_data[panel] = matrix
    panel_data[panel] = matrix.reindex(all_conds)

# Footer values come directly from the source macro tables, as in the original
# figure. They may describe a different fixed set of conditions from the wins.
def macro_series(panel, column):
    rows = macro_ci_all.loc[macro_ci_all["panel"].eq(panel)].set_index("Model")
    values = pd.to_numeric(rows[column], errors="coerce").reindex(m_order)
    if panel not in FIG1_OPTIONAL_PANELS and values.isna().any():
        raise ValueError(f"Missing {column} in {panel}: {values[values.isna()].index.tolist()}")
    return values

model_means = {panel: macro_series(panel, "estimate") for panel in panel_labels}
model_ci_lower = {panel: macro_series(panel, "ci_lower") for panel in panel_labels}
model_ci_upper = {panel: macro_series(panel, "ci_upper") for panel in panel_labels}
macro_eligible_condition_n = {
    panel: (int(n.min()) if len(n) else 0)
    for panel in panel_labels
    for n in [macro_series(panel, "fixed_condition_n").dropna()]
}
macro_ci_display_status = {
    panel: "source-defined macro estimates and CIs" for panel in panel_labels
}
print(f"Ready: {len(m_order)} models, {len(all_conds)} displayed conditions, "
      f"{len(panel_labels)} metric panels")


# =============================================================================
# Notebook cell 15
# =============================================================================

# Missing abbreviations fall back to shortened full condition names.
def normalise_condition_name(name):
    return " ".join(str(name).strip().lower().split())

abbreviation_map = {}
for path in (PROJECT_ROOT / "Other_data/con_abb.json",
             PROJECT_ROOT.parent / "Other_data/con_abb.json"):
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for row in json.load(handle):
                name = str(row.get("condition_name", "")).strip()
                abbreviation = str(row.get("abbreviation", "")).strip()
                if name and abbreviation:
                    abbreviation_map[normalise_condition_name(name)] = abbreviation
        print(f"Loaded {len(abbreviation_map)} abbreviations from {path}")
        break


# =============================================================================
# Notebook cell 18
# =============================================================================

# =============================================================================
# 6a. Presentation settings, win counts and display-ready matrices
# =============================================================================
# Nothing here changes which models, conditions or metric values enter Figure 1.
# Selection, eligibility and macro CIs come from the cells above. This cell only
# (i) orders and names columns, (ii) converts to display units, (iii) recomputes
# win counts with an explicit practical-equivalence margin, and (iv) prepares
# colours that are scaled within each panel and prevalence group.

import textwrap

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

required_objects = [
    "all_conds", "panel_labels", "panel_data", "full_panel_data",
    "panel_directions", "FIG1_PANEL_SPECS", "m_order",
    "model_means", "model_ci_lower", "model_ci_upper",
    "macro_eligible_condition_n", "npos_by_h",
    "common_conds", "rare_conds", "REF_HORIZON", "FIG1_MIN_TEST_EVENTS",
    "FIG1_EXCLUDED_CONDITION_PATTERNS",
]
missing_objects = [n for n in required_objects if n not in globals()]
if missing_objects:
    raise NameError("Missing Figure 1 objects:\n" + "\n".join(missing_objects))
if "abbreviation_map" not in globals():
    abbreviation_map = {}

REFERENCE_MODEL = globals().get("FIG1_REFERENCE_MODEL", "CRESCENT")
if REFERENCE_MODEL not in m_order:
    raise ValueError(f"Reference model {REFERENCE_MODEL!r} is absent.")

# ---- Editable presentation choices ------------------------------------------
# Columns are grouped by model family (thin gaps separate families). Models not
# listed are appended to the last non-reference group so nothing is dropped.
MODEL_GROUPS = [
    ("Survival baselines", ["Cox", "XGBoostSurvival"]),
    ("Deep survival",      ["Survtrace", "DDH"]),
    ("EHR sequence models", ["Motor", "SURVIVEHR", "Delphi-2M"]),
    ("Language model",     ["Gemma"]),
    ("Other",              ["New"]),
    ("Proposed",           [REFERENCE_MODEL]),
]
# Display names. "New" is a placeholder in the source notebook: set its real name.
MODEL_DISPLAY_NAMES = {
    "XGBoostSurvival": "XGBoost",
    "DynamicDeepHit": "DeepHit",
    "New": "New",   # TODO: replace with the manuscript name of this model
}
if MODEL_DISPLAY_NAMES.get("New") == "New" and "New" in m_order:
    print("NOTE: model 'New' still has its placeholder name; set "
          "MODEL_DISPLAY_NAMES['New'] before submission.")

# Display units, number formats and titles per metric column.
PANEL_DISPLAY = {
    "auroc": dict(scale=1.0, fmt=".2f", tick="%.2f"),
    "cindex_ipcw_cr": dict(scale=1.0, fmt=".2f", tick="%.2f"),
    "scaled_ibs_ipcw_ht": dict(scale=100.0, fmt=".1f", tick="%.1f"),
    "integrated_abs_calibration_error": dict(scale=1000.0, fmt=".1f", tick="%.1f"),
}
PANEL_TITLES = {
    "2-year AUROC": "2-year AUROC ↑",
    "5-year AUROC": "5-year AUROC ↑",
    "Scaled integrated Brier": f"Scaled IBS skill score ↑\n(%, {INTEGRATION_WINDOW_RECRUITMENT[0]}–{INTEGRATION_WINDOW_RECRUITMENT[1]} mo)",
    "Integrated calibration error": (
        f"Integrated calibration error ↓\n($\\times10^{{-3}}$, {INTEGRATION_WINDOW_RECRUITMENT[0]}–{INTEGRATION_WINDOW_RECRUITMENT[1]} mo)"
    ),
}
for _p, _s in FIG1_PANEL_SPECS.items():
    if _s["source"] == "cindex":
        PANEL_TITLES[_p] = "C-index (τ = 5 years) ↑"

# Smallest colour scale used per metric, so that tiny differences are not
# stretched to full colour: AUROC/C-index 0.01, IBS 0.002, calibration log2 0.25.
ADV_SCALE_FLOOR = {
    "auroc": 0.01, "cindex_ipcw_cr": 0.01,
    "scaled_ibs_ipcw_ht": 0.002, "integrated_abs_calibration_error": 0.25,
}
ADV_PERCENTILE = 95.0
LOW_EVENT_FLAG = 50   # rows with fewer 5-year test events get a dagger

# The main figure shows every panel, including the C-index. Because the
# supplement list is identical, only one figure is drawn. To move the C-index to
# a supplementary figure, remove it from FIG1_MAIN_PANELS.
FIG1_MAIN_PANELS = list(panel_labels)
FIG1_SUPPLEMENT_PANELS = list(FIG1_MAIN_PANELS)

# ---- Column order ------------------------------------------------------------
plot_order, group_of_model, group_names = [], {}, []
_listed = {m for _, ms in MODEL_GROUPS for m in ms}
_unlisted = [m for m in m_order if m not in _listed]
for gi, (gname, members) in enumerate(MODEL_GROUPS):
    members = [m for m in members if m in m_order]
    if gname == "Other":
        members = members + [m for m in _unlisted if m not in members]
    if not members:
        continue
    group_names.append(gname)
    for m in members:
        plot_order.append(m)
        group_of_model[m] = len(group_names) - 1
if sorted(plot_order) != sorted(m_order):
    raise RuntimeError("Model grouping dropped or duplicated a model.")
model_group_id = np.array([group_of_model[m] for m in plot_order])
display_name = {m: MODEL_DISPLAY_NAMES.get(m, m) for m in plot_order}

# ---- Pairwise comparison against the reference (complete-case) ----------------
# For every panel, CRESCENT is compared with each other model on ONE condition
# set: conditions where every model that reports the panel has a value, with at
# least WIN_MIN_EVENTS locked-test events at the panel's own horizon, HIV-related
# targets and conditions without rows for every model excluded.
# Per comparator: number of conditions where the reference is better / tied /
# worse, two-sided Wilcoxon signed-rank test on the paired condition-level
# differences (Holm-adjusted across comparators within a panel) and the mean
# rank of every model. Conditions are treated as exchangeable units, so these
# are descriptive; within-condition sampling uncertainty is in the macro CIs.
try:
    from scipy import stats as _stats
except ImportError:                                   # pragma: no cover
    _stats = None
    print("WARNING: scipy not found; Wilcoxon/Friedman p-values will be NaN.")

WIN_MIN_EVENTS = FIG1_MIN_TEST_EVENTS
HOLM_ALPHA = 0.05
PAIR_TIE_TOL = 1e-12
WIN_EXCLUSION_PATTERN = "|".join(f"(?:{p})" for p in FIG1_EXCLUDED_CONDITION_PATTERNS)
EPS_ERR = 1e-6
_missing_rows = set(globals().get("fig1_missing_row_conditions", []))


def events_at_horizon(horizon):
    """Locked-test events per condition at a horizon (reference model's table)."""
    if horizon in npos_by_h.columns:
        return pd.to_numeric(npos_by_h[horizon], errors="coerce")
    if "df_all" in globals():
        sub = df_all[df_all["section"].eq("within") & df_all["Model"].eq(REFERENCE_MODEL)
                     & df_all["horizon_months"].eq(horizon)]
        if len(sub):
            return sub.groupby("condition")["n_pos"].max().astype(float)
    print(f"WARNING: no event counts at {horizon} months; using {REF_HORIZON} months.")
    return pd.to_numeric(npos_by_h[REF_HORIZON], errors="coerce")


def holm_adjust(p):
    p = pd.Series(p, dtype=float)
    out = pd.Series(np.nan, index=p.index)
    valid = p.dropna().sort_values()
    m, running = len(valid), 0.0
    for rank, (idx, val) in enumerate(valid.items()):
        running = max(running, min(1.0, (m - rank) * val))
        out.loc[idx] = running
    return out


pairwise_table, mean_rank, complete_case_n = {}, {}, {}
friedman_p, complete_case_conditions, panel_models_used = {}, {}, {}
audit_frames = []
for panel in panel_labels:
    spec = FIG1_PANEL_SPECS[panel]
    column, direction = spec["column"], int(panel_directions[panel])
    horizon = int(spec.get("horizon_months", REF_HORIZON))
    matrix = full_panel_data[panel].reindex(columns=m_order).apply(pd.to_numeric, errors="coerce")
    present = [m for m in m_order if matrix[m].notna().any()]
    panel_models_used[panel] = present
    text = pd.Series(matrix.index.astype(str), index=matrix.index, dtype="string")
    events = events_at_horizon(horizon).reindex(matrix.index)
    keep = (
        ~text.str.contains(WIN_EXCLUSION_PATTERN, case=False, regex=True, na=False)
        & ~text.isin(list(_missing_rows))
        & (events >= int(WIN_MIN_EVENTS)).fillna(False)
        & matrix[present].notna().all(axis=1)
    )
    conds = matrix.index[keep.to_numpy()].tolist()
    complete_case_conditions[panel] = conds
    complete_case_n[panel] = len(conds)
    sub = matrix.loc[conds, present]

    if column == "integrated_abs_calibration_error":      # ratio scale
        oriented = -np.log(sub.clip(lower=EPS_ERR))
    else:
        oriented = sub * direction
    ranks = oriented.apply(lambda r: pd.Series(_stats.rankdata(-r.to_numpy(), method="average")
                                               if _stats else r.rank(ascending=False), index=r.index), axis=1)
    mr = ranks.mean(axis=0).reindex(m_order) if len(conds) else pd.Series(np.nan, index=m_order)
    mean_rank[panel] = mr
    friedman_p[panel] = np.nan
    if _stats is not None and len(present) >= 3 and len(conds) >= 3:
        try:
            friedman_p[panel] = float(_stats.friedmanchisquare(*[oriented[m].to_numpy() for m in present]).pvalue)
        except ValueError:
            pass

    rows = []
    for m in m_order:
        rec = dict(panel=panel, model=m, horizon_months=horizon, n_conditions=len(conds),
                   min_events=int(WIN_MIN_EVENTS), models_in_panel=len(present))
        if m == REFERENCE_MODEL or m not in present or not len(conds):
            rec.update(ref_better=np.nan, ties=np.nan, comparator_better=np.nan,
                       median_difference=np.nan, wilcoxon_p=np.nan)
        else:
            d = (oriented[REFERENCE_MODEL] - oriented[m]).to_numpy(float)   # > 0: reference better
            nz = d[np.abs(d) > PAIR_TIE_TOL]
            p = np.nan
            if _stats is not None and len(nz):
                try:
                    p = float(_stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided").pvalue)
                except ValueError:
                    p = np.nan
            rec.update(ref_better=int((d > PAIR_TIE_TOL).sum()), ties=int((np.abs(d) <= PAIR_TIE_TOL).sum()),
                       comparator_better=int((d < -PAIR_TIE_TOL).sum()),
                       median_difference=float(np.median(d)), wilcoxon_p=p)
        rec["mean_rank"] = float(mr.loc[m]) if np.isfinite(mr.loc[m]) else np.nan
        rec["complete_case_mean"] = float(sub[m].mean()) if (m in present and len(conds)) else np.nan
        rows.append(rec)
    tab = pd.DataFrame(rows).set_index("model")
    tab["holm_p"] = holm_adjust(tab["wilcoxon_p"])
    tab["significant_direction"] = np.where(
        tab["holm_p"] < HOLM_ALPHA, np.where(tab["median_difference"] > 0, 1, -1), 0)
    tab.loc[tab["holm_p"].isna(), "significant_direction"] = 0
    tab["friedman_p"] = friedman_p[panel]
    pairwise_table[panel] = tab
    audit_frames.append(tab.reset_index())

pairwise_audit_df = pd.concat(audit_frames, ignore_index=True)
pairwise_audit_df.to_csv("Figure1_pairwise_comparison_audit.csv", index=False)

# ---- Display matrices and group-wise colour scaling ---------------------------
n_common = len(common_conds)
row_group = np.array([0] * n_common + [1] * len(rare_conds))
EPS_ERR = 1e-6
display_matrix, advantage_scaled = {}, {}
macro_display = {}
for panel in panel_labels:
    column = FIG1_PANEL_SPECS[panel]["column"]
    scale = PANEL_DISPLAY[column]["scale"]
    raw = (panel_data[panel].reindex(index=all_conds, columns=plot_order)
           .apply(pd.to_numeric, errors="coerce"))
    display_matrix[panel] = raw * scale
    ref = raw[REFERENCE_MODEL]
    if column == "integrated_abs_calibration_error":
        # Ratio scale: positive = lower error than CRESCENT.
        adv = -np.log2(raw.clip(lower=EPS_ERR).div(ref.clip(lower=EPS_ERR), axis=0))
    else:
        adv = raw.sub(ref, axis=0) * int(panel_directions[panel])
    scaled = adv.copy()
    for g in (0, 1):
        block = adv.iloc[np.where(row_group == g)[0]].to_numpy(float)
        finite = np.abs(block[np.isfinite(block)])
        s = float(np.percentile(finite, ADV_PERCENTILE)) if len(finite) else 1.0
        s = max(s, ADV_SCALE_FLOOR[column])
        scaled.iloc[np.where(row_group == g)[0]] = (block / s).clip(-1, 1)
    advantage_scaled[panel] = scaled
    macro_display[panel] = {
        "est": pd.to_numeric(model_means[panel].reindex(plot_order), errors="coerce") * scale,
        "lo": pd.to_numeric(model_ci_lower[panel].reindex(plot_order), errors="coerce") * scale,
        "hi": pd.to_numeric(model_ci_upper[panel].reindex(plot_order), errors="coerce") * scale,
    }

advantage_cmap = LinearSegmentedColormap.from_list(
    "advantage", ["#b2182b", "#e08a73", "#f6f6f6", "#7fb2d6", "#2166ac"], N=256)
advantage_cmap.set_bad("#ececec")

def short(name, width=22):
    key = " ".join(str(name).strip().lower().split())
    if key in abbreviation_map:
        return abbreviation_map[key]
    name = str(name).strip()
    return name if len(name) <= width else name[: width - 1] + "…"

def event_count(condition):
    if condition not in npos_by_h.index or 60 not in npos_by_h.columns:
        return None
    v = npos_by_h.loc[condition, 60]
    return None if pd.isna(v) else int(v)

def fmt_value(v, fmt, strip_zero=False):
    """Compact cell text: no "-0.0", whole numbers from 10 up, optional ".73" style."""
    if fmt.endswith("f") and abs(v) >= 10:
        fmt = ".0f"
    s = format(v, fmt)
    if float(s) == 0.0:
        s = format(0.0, fmt)
    if strip_zero:
        if s.startswith("0."):
            s = s[1:]
        elif s.startswith("-0."):
            s = "-" + s[2:]
    return s


def wrap_to_width(text, width_in, fontsize, style="normal"):
    """Greedy wrap using the real rendered width of the current font."""
    probe = plt.figure(figsize=(1, 1))
    renderer = probe.canvas.get_renderer()
    lines, current = [], ""
    for word in text.split():
        trial = (current + " " + word).strip()
        t = probe.text(0, 0, trial, fontsize=fontsize, style=style)
        too_wide = t.get_window_extent(renderer).width / probe.dpi > width_in
        t.remove()
        if too_wide and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    plt.close(probe)
    return lines

print(f"Ready: {len(plot_order)} models in {len(group_names)} groups, "
      f"{len(all_conds)} rows; main panels={FIG1_MAIN_PANELS}")
for panel in panel_labels:
    tab = pairwise_table[panel]
    print(f"  {panel}: complete-case N={complete_case_n[panel]} "
          f"(models={len(panel_models_used[panel])}); Friedman p={friedman_p[panel]:.3g}; "
          f"{REFERENCE_MODEL} significantly better than "
          f"{int((tab['significant_direction'] == 1).sum())}, worse than "
          f"{int((tab['significant_direction'] == -1).sum())} models")
print("Saved Figure1_pairwise_comparison_audit.csv")


# =============================================================================
# Notebook cell 19
# =============================================================================

# =============================================================================
# 6b. Drawing routine (sizes are in inches so the layout is exact)
# =============================================================================
CELL_W, CELL_H = 0.148, 0.166          # one heatmap cell
GROUP_GAP_U = 0.30                     # gap between model families, in cells
ROW_GAP_U = 0.42                       # gap between common and rare blocks
PANEL_GAP = 0.10                       # gap between panels sharing a y-axis
PANEL_GAP_OWN_AXIS = 0.34              # gap before a strip that needs its own axis
LEFT_MARGIN, RIGHT_MARGIN = 1.10, 0.42
NAVY, INK, MUTED = "#1f4e79", "#1b1b1b", "#5f6670"
ROW_SIG_Y, ROW_COUNT_Y, ROW_RANK_Y = 0.88, 0.56, 0.24   # vertical positions of the three comparison rows
FIG1_TIFF_DPI = 600
FIG1_SHOW_FOOTNOTE = False   # True prints the explanatory text under the figure; a draft caption file is always saved
LABEL_ANGLE = 45                       # tilt of model names (top and bottom)
PREV_LABEL_X = 0.30                    # inches from the left edge for the rotated "10 most/least prevalent" labels (was 0.14)
COND_HEADER_X = LEFT_MARGIN - 0.02     # right edge of the "Condition (5-year test events)" header (was LEFT_MARGIN - 0.10)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "mathtext.default": "regular", "axes.linewidth": 0.5,
})

xpos = np.arange(len(plot_order)) + GROUP_GAP_U * model_group_id
ypos = np.arange(len(all_conds)) + np.where(row_group == 1, ROW_GAP_U, 0.0)
X_LIM = (-0.5, xpos[-1] + 0.5)
Y_LIM = (ypos[-1] + 0.5, -0.5)
REF_INDEX = plot_order.index(REFERENCE_MODEL)


def _axis_key(panel):
    return "discrimination" if FIG1_PANEL_SPECS[panel]["source"] in ("per_condition", "cindex") else panel


def _strip_limits(panels):
    """Shared limits for discrimination strips; own limits for IBS/calibration."""
    limits, groups = {}, {}
    for p in panels:
        groups.setdefault(_axis_key(p), []).append(p)
    for key, members in groups.items():
        vals = np.concatenate([
            np.concatenate([macro_display[p]["lo"].to_numpy(float),
                            macro_display[p]["hi"].to_numpy(float),
                            macro_display[p]["est"].to_numpy(float)])
            for p in members])
        vals = vals[np.isfinite(vals)]
        lo, hi = float(vals.min()), float(vals.max())
        span = max(hi - lo, 1e-9)
        lo, hi = lo - 0.18 * span, hi + 0.34 * span      # head-room for value labels
        if key == "discrimination":
            hi = min(hi, 1.0)
        else:
            bl = FIG1_PANEL_SPECS[members[0]].get("bounds", (None, None))[0]
            if bl is not None:
                lo = max(lo, float(bl))
        for p in members:
            limits[p] = (lo, hi)
    return limits


def draw_figure(panels, stem, title=None, letters=True):
    n_models, n_rows = len(plot_order), len(all_conds)
    panel_w = (X_LIM[1] - X_LIM[0]) * CELL_W
    heat_h = (Y_LIM[0] - Y_LIM[1]) * CELL_H

    # horizontal placement
    lefts, x = [], LEFT_MARGIN
    for k, p in enumerate(panels):
        if k:
            own = _axis_key(p) != _axis_key(panels[k - 1]) or _axis_key(p) == p
            x += PANEL_GAP_OWN_AXIS if own else PANEL_GAP
        lefts.append(x)
        x += panel_w
    fig_w = x + RIGHT_MARGIN

    # footnote (wrapped to the figure width) decides the bottom margin
    n_cc = sorted({complete_case_n[p] for p in panels})
    n_text = str(n_cc[0]) if len(n_cc) == 1 else f"{n_cc[0]}\u2013{n_cc[-1]}"
    note = (
        f"Heatmap colour shows each model\u2019s difference from {REFERENCE_MODEL} in the direction of improvement "
        "(calibration: log\u2082 error ratio), scaled to the 95th percentile within each panel and prevalence block; "
        f"{REFERENCE_MODEL} (last column) is the reference. Rows are the 10 most and 10 least prevalent eligible conditions, "
        f"ordered by overall cohort prevalence; n = locked-test events at 5 years. "
        f"Comparison rows use one complete-case set per panel (N = {n_text}; every model evaluated, \u2265{int(FIG1_MIN_TEST_EVENTS)} "
        "locked-test events at the panel\u2019s horizon, HIV-related targets excluded). "
        f"Symbols: \u25b2 {REFERENCE_MODEL} significantly better than the column model, \u25bc column model significantly better, "
        f"ns not significant (two-sided Wilcoxon signed-rank test on paired condition-level differences, Holm-adjusted within panel, "
        f"p < {HOLM_ALPHA:g}); the count is the number of those conditions on which {REFERENCE_MODEL} outperforms the column model. "
        "Mean rank is taken across the same conditions (1 = best; ties averaged; best shaded). "
        "Conditions are treated as exchangeable units, so tests are descriptive. "
        f"Lower strips: macro-average from each model\u2019s source table with participant-bootstrap 95% CI; "
        f"the shaded band is the {REFERENCE_MODEL} CI."
    )
    _extra = globals().get("fig1_extra_row_conditions", [])
    if _extra:
        note += (f" {len(_extra)} condition(s) modelled by comparators but not by {REFERENCE_MODEL} were ignored; "
                 "macro summaries of comparators may include them.")
    _dropped = globals().get("fig1_missing_row_conditions", [])
    if _dropped:
        note += (f" {len(_dropped)} condition(s) without results for every model were excluded for all models "
                 "(see Figure1_conditions_dropped_missing_model_rows.csv).")
    text_w = fig_w - 0.15
    note_lines = wrap_to_width(note, text_w, 5.0)
    families = "   |   ".join(
        f"{g}: " + ", ".join(display_name[m] for m in plot_order if group_names[group_of_model[m]] == g)
        for g in group_names)
    fam_lines = wrap_to_width("Columns, left to right — " + families, text_w, 5.0, style="italic")

    # vertical placement (inches, measured from the bottom)
    PT = 1.0 / 72
    h_top_pad, h_title, h_labels = 0.06, 0.34 + (0.06 if title else 0.0), 0.50
    h_wins, h_gap1, h_strip, h_xlab, h_gap2 = 1.12, 0.14, 1.00, 0.46, 0.04
    h_cbar = 0.50
    h_note = ((len(note_lines) + len(fam_lines)) * 7.4 * PT + 0.14) if FIG1_SHOW_FOOTNOTE else 0.14
    fig_h = (h_top_pad + h_title + h_labels + heat_h + 0.07 + h_wins + h_gap1
             + h_strip + h_xlab + h_gap2 + h_cbar + h_note)
    y = fig_h - h_top_pad
    y_title_base = y - h_title
    y_heat_top = y_title_base - h_labels
    y_heat_bot = y_heat_top - heat_h
    y_wins_bot = y_heat_bot - 0.07 - h_wins
    y_strip_top = y_wins_bot - h_gap1
    y_strip_bot = y_strip_top - h_strip
    y_cbar_bot = y_strip_bot - h_xlab - h_gap2 - h_cbar

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    def add_ax(x0, y0, w, h):
        a = fig.add_axes([x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h])
        a.patch.set_alpha(0.0)
        return a

    def ftext(x0, y0, s, **kw):
        return fig.text(x0 / fig_w, y0 / fig_h, s, **kw)

    if title:
        ftext(0.05, fig_h - 0.14, title, fontsize=8.5, fontweight="bold", color=INK, va="center")

    strip_limits = _strip_limits(panels)

    # ---- row labels --------------------------------------------------------
    ax_rows = add_ax(0.0, y_heat_bot, LEFT_MARGIN - 0.08, heat_h)
    ax_rows.set_xlim(0, 1); ax_rows.set_ylim(*Y_LIM); ax_rows.axis("off")
    for i, condition in enumerate(all_conds):
        n_events = event_count(condition)
        n_text_i = "–" if n_events is None else f"{n_events:,}"
        ax_rows.text(1.0, ypos[i], f"{short(condition)}  ({n_text_i})",
                     ha="right", va="center", fontsize=5.9, color=INK)
    for label, rows in (("10 most prevalent", range(0, n_common)),
                        ("10 least prevalent", range(n_common, n_rows))):
        mid = float(np.mean([ypos[i] for i in rows]))
        ax_rows.text(PREV_LABEL_X / (LEFT_MARGIN - 0.08), mid, label, rotation=90, ha="center", va="center",
                     fontsize=6.4, fontweight="bold", color=MUTED, clip_on=False)
    ftext(COND_HEADER_X, y_heat_top + h_labels - 0.02,
          "Condition\n(5-year test events)", ha="right", va="top",
          fontsize=6.2, fontweight="bold", color=MUTED, linespacing=1.15)

    # ---- heatmaps, wins and strips -----------------------------------------
    for k, panel in enumerate(panels):
        spec = FIG1_PANEL_SPECS[panel]
        fmt = PANEL_DISPLAY[spec["column"]]["fmt"]
        ax = add_ax(lefts[k], y_heat_bot, panel_w, heat_h)
        ax.set_xlim(*X_LIM); ax.set_ylim(*Y_LIM); ax.axis("off")
        values = display_matrix[panel].to_numpy(float)
        colours = advantage_scaled[panel].to_numpy(float)
        for i in range(n_rows):
            for j in range(n_models):
                c = colours[i, j]
                face = advantage_cmap(0.5 * (c + 1.0)) if np.isfinite(c) else advantage_cmap.get_bad()
                ax.add_patch(Rectangle((xpos[j] - 0.5, ypos[i] - 0.5), 1, 1,
                                       facecolor=face, edgecolor="white", linewidth=0.5))
                if np.isfinite(values[i, j]):
                    ax.text(xpos[j], ypos[i], fmt_value(values[i, j], fmt, strip_zero=spec["column"] in ("auroc", "cindex_ipcw_cr")), ha="center", va="center",
                            fontsize=4.7, color="white" if (np.isfinite(c) and abs(c) > 0.72) else INK)
        for j, m in enumerate(plot_order):
            is_ref = m == REFERENCE_MODEL
            ax.text(xpos[j] - 0.2, Y_LIM[1] - 0.3, display_name[m], rotation=LABEL_ANGLE,
                    rotation_mode="anchor", ha="left", va="bottom",
                    fontsize=5.8, color=INK, clip_on=False)
        top_y = y_heat_top + h_labels
        if letters:
            ftext(lefts[k], top_y + 0.30, "ABCDEFG"[k], fontsize=10, fontweight="normal",
                  color=INK, ha="left", va="top")
        ftext(lefts[k] + (0.20 if letters else 0.0), top_y + 0.295, PANEL_TITLES.get(panel, panel),
              fontsize=6.4, fontweight="bold", color="#263746", ha="left", va="top", linespacing=1.15)

        # pairwise comparison rows: significance symbol, count, mean rank
        axw = add_ax(lefts[k], y_wins_bot, panel_w, h_wins)
        axw.set_xlim(*X_LIM); axw.set_ylim(0, 1); axw.axis("off")
        tab, ranks = pairwise_table[panel], mean_rank[panel]
        best_rank = np.nanmin(ranks.to_numpy(float)) if np.isfinite(ranks.to_numpy(float)).any() else np.nan
        for j, m in enumerate(plot_order):
            if m == REFERENCE_MODEL:
                for yy in (ROW_SIG_Y, ROW_COUNT_Y):
                    axw.text(xpos[j], yy, "\u2014", ha="center", va="center", fontsize=5.4, color=MUTED)
            elif np.isfinite(tab.loc[m, "ref_better"]):
                direction_sig = int(tab.loc[m, "significant_direction"])
                if direction_sig == 1:
                    axw.text(xpos[j], ROW_SIG_Y, "\u25b2", ha="center", va="center", fontsize=6.4, color=INK)
                elif direction_sig == -1:
                    axw.text(xpos[j], ROW_SIG_Y, "\u25bc", ha="center", va="center", fontsize=6.4, color=INK)
                else:
                    axw.text(xpos[j], ROW_SIG_Y, "ns", ha="center", va="center", fontsize=4.8, color=MUTED)
                axw.text(xpos[j], ROW_COUNT_Y, f"{int(tab.loc[m, 'ref_better'])}", ha="center", va="center",
                         fontsize=5.0, color=INK)
            else:
                for yy in (ROW_SIG_Y, ROW_COUNT_Y):
                    axw.text(xpos[j], yy, "n/a", ha="center", va="center", fontsize=4.6, color=MUTED)
            r = ranks.get(m, np.nan)
            if np.isfinite(r):
                best = np.isfinite(best_rank) and abs(r - best_rank) < 1e-9
                axw.text(xpos[j], ROW_RANK_Y, f"{r:.1f}", ha="center", va="center", fontsize=5.0, color=INK,
                         bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.4", facecolor="#dcdcdc",
                                   edgecolor="none") if best else None)

        # macro strip
        axs = add_ax(lefts[k], y_strip_bot, panel_w, h_strip)
        d = macro_display[panel]
        lo_lim, hi_lim = strip_limits[panel]
        axs.set_xlim(*X_LIM); axs.set_ylim(lo_lim, hi_lim)
        ref_lo, ref_hi, ref_est = (d["lo"].iloc[REF_INDEX], d["hi"].iloc[REF_INDEX], d["est"].iloc[REF_INDEX])
        if np.isfinite(ref_lo) and np.isfinite(ref_hi):
            axs.axhspan(ref_lo, ref_hi, color="#8a8f98", alpha=0.13, linewidth=0, zorder=0)
        est, lo, hi = d["est"].to_numpy(float), d["lo"].to_numpy(float), d["hi"].to_numpy(float)
        for j in range(n_models):
            if not np.isfinite(est[j]):
                continue
            is_ref = j == REF_INDEX
            colour = "#111111" if is_ref else "#6b717b"
            if np.isfinite(lo[j]) and np.isfinite(hi[j]):
                axs.plot([xpos[j]] * 2, [lo[j], hi[j]], color=colour, linewidth=0.9, solid_capstyle="butt", zorder=4)
                for cap in (lo[j], hi[j]):
                    axs.plot([xpos[j] - 0.18, xpos[j] + 0.18], [cap, cap], color=colour, linewidth=0.9, zorder=4)
            axs.scatter([xpos[j]], [est[j]], s=15 if is_ref else 10, color=colour, edgecolors="white",
                        linewidths=0.4, zorder=5)
            top = hi[j] if np.isfinite(hi[j]) else est[j]
            axs.text(xpos[j], top + 0.025 * (hi_lim - lo_lim), fmt_value(est[j], fmt, strip_zero=spec["column"] in ("auroc", "cindex_ipcw_cr")),
                     ha="center", va="bottom", fontsize=4.6, color=INK if is_ref else MUTED, zorder=6)
        axs.set_xticks(xpos)
        axs.set_xticklabels([display_name[m] for m in plot_order], rotation=LABEL_ANGLE, ha="right", va="top",
                            rotation_mode="anchor", fontsize=5.8)
        axs.tick_params(axis="x", length=0, pad=2)
        axs.grid(axis="y", color="#e9e9e9", linewidth=0.5, zorder=0)
        axs.set_axisbelow(True)
        for side in ("top", "right"):
            axs.spines[side].set_visible(False)
        axs.spines["bottom"].set_color("#b5b5b5")
        own_axis = k == 0 or _axis_key(panel) != _axis_key(panels[k - 1]) or _axis_key(panel) == panel
        axs.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        if own_axis:
            axs.spines["left"].set_color("#b5b5b5")
            axs.yaxis.set_major_formatter(ticker.FormatStrFormatter(PANEL_DISPLAY[spec["column"]]["tick"]))
            axs.tick_params(axis="y", labelsize=5.4, length=2, pad=1.5, color="#b5b5b5")
            if k == 0:
                axs.set_ylabel("Macro-average (95% CI)", fontsize=5.8, color=MUTED, labelpad=3)
        else:
            axs.spines["left"].set_visible(False)
            axs.tick_params(axis="y", left=False, labelleft=False)

    for yy, lab in ((ROW_SIG_Y, f"Significance\nvs {REFERENCE_MODEL}"),
                    (ROW_COUNT_Y, f"{REFERENCE_MODEL} better\non n conditions"),
                    (ROW_RANK_Y, "Mean rank\n(1 = best)")):
        ftext(LEFT_MARGIN - 0.10, y_wins_bot + h_wins * yy, lab, ha="right", va="center",
              fontsize=5.7, fontweight="bold", color=MUTED, linespacing=1.15)

    # ---- colour key ---------------------------------------------------------
    axc = add_ax(LEFT_MARGIN, y_cbar_bot + 0.34, 1.6, 0.07)
    axc.imshow(np.linspace(-1, 1, 256)[None, :], aspect="auto", cmap=advantage_cmap, extent=[-1, 1, 0, 1])
    axc.set_yticks([]); axc.set_xticks([])
    for s in axc.spines.values():
        s.set_linewidth(0.4); s.set_color("#999999")
    for xx, lab, ha in ((-1, "Worse", "left"), (0, "Equal", "center"), (1, "Better", "right")):
        axc.text(xx, -0.35, lab, ha=ha, va="top", fontsize=5.4, color=MUTED, transform=axc.transData)
    cbar_note = ftext(LEFT_MARGIN + 1.6 + 0.10, y_cbar_bot + 0.34 + 0.035, f"relative to {REFERENCE_MODEL} (same metric, same prevalence block)",
          fontsize=5.4, color=MUTED, ha="left", va="center")
    # arrow key placed just after the colour-bar note, so it follows the note whatever its length
    axc.annotate("\u2191 higher is better    \u2193 lower is better", xy=(1.0, 0.5), xycoords=cbar_note,
                 xytext=(14, 0), textcoords="offset points", ha="left", va="center",
                 fontsize=5.4, color=INK, annotation_clip=False)

    ftext(LEFT_MARGIN, y_cbar_bot + 0.07,
          f"\u25b2 {REFERENCE_MODEL} significantly better      \u25bc column model significantly better      "
          "ns not significant      (Wilcoxon signed-rank on paired conditions, Holm-adjusted p < "
          f"{HOLM_ALPHA:g})      shaded: best mean rank",
          fontsize=5.6, color=INK, ha="left", va="center")

    # ---- footnotes (optional) and draft caption ---------------------------
    if FIG1_SHOW_FOOTNOTE:
        y_note = y_cbar_bot - 0.06
        for line in note_lines:
            ftext(0.05, y_note, line, fontsize=5.0, color=MUTED, ha="left", va="top")
            y_note -= 7.4 * PT
        for line in fam_lines:
            ftext(0.05, y_note, line, fontsize=5.0, color=MUTED, ha="left", va="top", style="italic")
            y_note -= 7.4 * PT
    with open(f"{stem}_caption_draft.txt", "w", encoding="utf-8") as fh:
        fh.write(" ".join(note_lines) + "\n\n" + " ".join(fam_lines) + "\n")

    exports = (
        ("tiff", {"dpi": FIG1_TIFF_DPI, "pil_kwargs": {"compression": "tiff_lzw"}}),
        ("pdf", {}),
        ("svg", {}),
    )
    for ext, kw in exports:
        fig.savefig(f"{stem}.{ext}", facecolor="white", **kw)
    print(f"Saved {stem}.tiff / .pdf / .svg   ({fig_w:.2f} x {fig_h:.2f} in)")
    return fig


# =============================================================================
# Notebook cell 20
# =============================================================================

# =============================================================================
# 6c. Render the main figure and the supplementary all-metrics figure
# =============================================================================
FIG1_OUTPUT_DIR = OUTPUT_DIR
FIG1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG1_MAIN_STEM = FIG1_OUTPUT_DIR / "figure2"
FIG1_SUPP_STEM = FIG1_OUTPUT_DIR / "Figure1_supplement_all_metrics"

fig_main = draw_figure(FIG1_MAIN_PANELS, FIG1_MAIN_STEM)
plt.show()

if FIG1_SUPPLEMENT_PANELS != FIG1_MAIN_PANELS:
    fig_supp = draw_figure(FIG1_SUPPLEMENT_PANELS, FIG1_SUPP_STEM)
    plt.show()

print("\nPairwise comparison audit saved to Figure1_pairwise_comparison_audit.csv.")
