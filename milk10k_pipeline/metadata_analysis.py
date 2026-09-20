from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------------------------------------------------------
# Column survey
# ---------------------------------------------------------------------------

def survey_columns(
    df: pd.DataFrame,
    id_cols: List[str],
    target_col: str,
) -> pd.DataFrame:
    exclude = set(id_cols) | {target_col}
    rows = []
    for col in df.columns:
        if col in exclude:
            continue
        series = df[col]
        pct_missing = series.isna().mean() * 100
        n_unique = series.nunique(dropna=True)

        if pd.api.types.is_bool_dtype(series):
            inferred = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            inferred = "numeric"
        elif n_unique <= max(20, int(0.05 * len(series))):
            inferred = "categorical"
        else:
            inferred = "free_text"

        rows.append({
            "column": col,
            "inferred_type": inferred,
            "pct_missing": round(pct_missing, 1),
            "n_unique": n_unique,
        })

    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Categorical fields vs target
# ---------------------------------------------------------------------------

def categorical_vs_target(
    df: pd.DataFrame,
    field: str,
    target_col: str,
    plot: bool = True,
    figsize=(10, 6),
):

    subset = df[[field, target_col]].dropna()
    contingency = pd.crosstab(subset[field], subset[target_col])
    row_normalized = contingency.div(contingency.sum(axis=1), axis=0)

    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    verdict = "informative" if p_value < 0.05 else "not clearly informative"

    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        row_normalized.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(f"{target_col} distribution within each '{field}' category\n"
                     f"(chi2={chi2:.1f}, p={p_value:.4g} -> {verdict})")
        ax.set_xlabel(field)
        ax.set_ylabel("Proportion")
        ax.legend(title=target_col, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

    return {
        "table": row_normalized,
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Numeric fields vs target
# ---------------------------------------------------------------------------

def numeric_vs_target(
    df: pd.DataFrame,
    field: str,
    target_col: str,
    plot: bool = True,
    figsize=(10, 6),
):
    subset = df[[field, target_col]].dropna()
    groups = [g[field].values for _, g in subset.groupby(target_col)]
    groups = [g for g in groups if len(g) > 1]  # ANOVA needs >1 sample per group

    f_stat, anova_p = stats.f_oneway(*groups)
    kw_stat, kw_p = stats.kruskal(*groups)

    verdict = "informative" if anova_p < 0.05 else "not clearly informative"

    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        subset.boxplot(column=field, by=target_col, ax=ax, rot=90)
        ax.set_title(f"{field} by {target_col}\n"
                     f"(ANOVA F={f_stat:.2f}, p={anova_p:.4g} -> {verdict})")
        plt.suptitle("")
        ax.set_xlabel(target_col)
        ax.set_ylabel(field)
        plt.tight_layout()

    return {
        "f_stat": f_stat,
        "anova_p": anova_p,
        "kw_stat": kw_stat,
        "kw_p": kw_p,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Summarize which fields matter most
# ---------------------------------------------------------------------------

def summarize_associations(results: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for field, res in results.items():
        p = res.get("p_value", res.get("anova_p"))
        kind = "categorical (chi2)" if "chi2" in res else "numeric (ANOVA)"
        rows.append({
            "field": field,
            "test": kind,
            "p_value": p,
            "verdict": res["verdict"],
        })
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
