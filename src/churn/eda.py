"""Basic EDA - dtypes, missing values, class balance, churn by segment.

usage: python -m churn.eda
"""
from __future__ import annotations

import pandas as pd

from . import config
from .data import clean


def main() -> None:
    raw = pd.read_csv(config.DATA_PATH)
    lines = ["# EDA summary", f"Shape: {raw.shape}", ""]

    lines += ["## Raw dtypes", "```", raw.dtypes.to_string(), "```"]
    blank_tc = (raw["TotalCharges"].astype(str).str.strip() == "").sum()
    lines += [f"TotalCharges is parsed as **{raw['TotalCharges'].dtype}**; "
              f"{blank_tc} rows contain blank strings.", ""]

    df = clean(raw)
    na = df.isna().sum()
    lines += ["## Missing after type coercion", "```", na[na > 0].to_string() or "none", "```"]
    lines += [f"All blank TotalCharges rows have tenure==0: "
              f"{bool((df.loc[df.TotalCharges.isna(), 'tenure'] == 0).all())}", ""]

    vc = df[config.TARGET].value_counts(normalize=True).round(3)
    lines += ["## Target balance", "```", vc.to_string(), "```", ""]

    lines += ["## Numeric summary", "```",
              df[config.NUMERIC_FEATURES].describe().round(2).to_string(), "```", ""]

    y = (df[config.TARGET] == "Yes").astype(int)
    lines += ["## Churn rate by key categoricals"]
    for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport"]:
        rate = y.groupby(df[col]).mean().round(3).sort_values(ascending=False)
        lines += [f"**{col}**", "```", rate.to_string(), "```"]

    text = "\n".join(lines)
    config.REPORT_DIR.mkdir(exist_ok=True)
    (config.REPORT_DIR / "eda_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
