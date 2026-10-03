"""Tiny helpers to write tables into Markdown reports without extra dependencies."""
from __future__ import annotations

import pandas as pd


def df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a GitHub Markdown table (no `tabulate` dependency)."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        cells = [str(v).replace("|", "/").replace("\n", " ") for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
