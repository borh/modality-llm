"""
Analyze the effect of each augmentation strategy on model predictions.
"""

from typing import Callable

import altair as alt
import polars as pl


def collect_modal_augmentation(
    examples: list[dict],
    classify_fn: Callable[[str], dict[str, int]],
    substitute_variants_fn: Callable[[dict], list[dict]],
) -> pl.DataFrame:
    """
    For each example, generate substitution variants, classify both original
    and variant, and record:
      - strategy
      - orig_category
      - preserved (bool)
      - confidence_drop (float)
    Returns a Polars DataFrame ready for grouping.
    """
    rows = []
    for ex in examples:
        original = ex["utt"]
        orig_dist = classify_fn(original)
        total = sum(orig_dist.values()) or 1
        orig_conf = max(orig_dist.values()) / total if orig_dist else 0.0
        orig_pred = max(orig_dist.items(), key=lambda x: x[1])[0] if orig_dist else None

        # keep track of the original (gold) category
        orig_cat = ex.get("res")
        if isinstance(orig_cat, list):
            orig_cat = orig_cat[0] if orig_cat else None
        for var in substitute_variants_fn(ex):
            strat = var["transformation_strategy"]
            utt = var["utt"]
            dist = classify_fn(utt)
            tot = sum(dist.values()) or 1
            conf = max(dist.values()) / tot if dist else 0.0
            pred = max(dist.items(), key=lambda x: x[1])[0] if dist else None
            rows.append(
                {
                    "strategy": strat,
                    "orig_category": orig_cat,
                    "preserved": pred == orig_pred,
                    "confidence_drop": orig_conf - conf,
                }
            )
    # always return a DF with the four columns, even if empty
    if not rows:
        return pl.DataFrame(
            {
                "strategy": [],
                "orig_category": [],
                "preserved": [],
                "confidence_drop": [],
            }
        )
    return pl.DataFrame(rows)


def bootstrap_ci(arr, n_boot=1000, ci=0.95):
    """Compute bootstrap CI on a 1d array of floats."""
    import numpy as np

    arr = np.array(arr, dtype=float)
    boots = [
        np.mean(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def analyze_augmentation_effects(df: pl.DataFrame):
    """Return three DataFrames: overall, by‐category, and raw‐drop for violin."""
    # Use list-of-dicts directly
    pdf = df.to_dicts()
    # 1) overall per‐strategy
    from collections import defaultdict

    import numpy as np

    # Group by strategy
    strat_groups = defaultdict(list)
    for row in pdf:
        strat_groups[row["strategy"]].append(row)
    overall = []
    for strat, grp in strat_groups.items():
        pres = [r["preserved"] for r in grp]
        drops = [r["confidence_drop"] for r in grp]
        lo_p, hi_p = bootstrap_ci(pres)
        lo_d, hi_d = bootstrap_ci(drops)
        overall.append(
            {
                "strategy": strat,
                "pres_rate": np.mean(pres) if pres else 0.0,
                "pres_ci_lo": lo_p,
                "pres_ci_hi": hi_p,
                "avg_drop": np.mean(drops) if drops else 0.0,
                "drop_ci_lo": lo_d,
                "drop_ci_hi": hi_d,
            }
        )
    overall_df = pl.DataFrame(overall)
    # 2) by strategy × category
    bycat_groups = defaultdict(list)
    for row in pdf:
        bycat_groups[(row["strategy"], row["orig_category"])].append(row)
    bycat = []
    for (strat, cat), grp in bycat_groups.items():
        pres = [r["preserved"] for r in grp]
        lo_p, hi_p = bootstrap_ci(pres)
        bycat.append(
            {
                "strategy": strat,
                "orig_category": cat,
                "pres_rate": np.mean(pres) if pres else 0.0,
                "pres_ci_lo": lo_p,
                "pres_ci_hi": hi_p,
            }
        )
    bycat_df = pl.DataFrame(bycat)
    # 3) raw for violin (rename drop)
    violin_df = df.select(
        [
            pl.col("strategy"),
            pl.col("confidence_drop").alias("drop"),
        ]
    )
    return overall_df, bycat_df, violin_df


def plot_aug_dashboard(
    overall_df: pl.DataFrame,
    bycat_df: pl.DataFrame,
    violin_df: pl.DataFrame,
    model_name: str,
    taxonomy: str,
):
    """Build an Altair dashboard: bars+EB, heatmap, faceted violin+jitter."""
    import altair as alt

    # Altair needs a Pandas DataFrame to infer dtypes
    odf = overall_df.to_pandas()
    bdf = bycat_df.to_pandas()
    vdf = violin_df.to_pandas()
    # Overall bar + errorbars
    bar = (
        alt.Chart(odf)
        .mark_bar()
        .encode(
            x=alt.X("strategy:N", title="Strategy"),
            y=alt.Y("pres_rate:Q", title="Preservation Rate"),
            tooltip=["strategy", "pres_rate", "pres_ci_lo", "pres_ci_hi"],
        )
    )
    eb = (
        alt.Chart(odf)
        .mark_errorbar()
        .encode(
            x="strategy:N",
            y="pres_rate:Q",
            yError="pres_ci_hi:Q",
            yError2="pres_ci_lo:Q",
        )
    )
    # Heatmap by category
    heat = (
        alt.Chart(bdf)
        .mark_rect()
        .encode(
            x=alt.X("orig_category:N", title="Original Category"),
            y=alt.Y("strategy:N", title="Strategy"),
            color=alt.Color("pres_rate:Q", scale=alt.Scale(scheme="blues")),
            tooltip=[
                "orig_category",
                "strategy",
                "pres_rate",
                "pres_ci_lo",
                "pres_ci_hi",
            ],
        )
    )
    # Violin + jitter overlay across all strategies (color‐coded on one chart)
    vd = violin_df.to_dicts()
    drops = [row["drop"] for row in vd] if vd else [0.0, 1.0]
    extent = [min(drops), max(drops)] if drops else [0.0, 1.0]
    density = (
        alt.Chart(vdf)
        .transform_density(
            "drop",
            as_=["drop", "density"],
            extent=extent,
            groupby=["strategy"],
        )
        .mark_area(orient="horizontal", opacity=0.3)
        .encode(
            alt.X(
                "density:Q",
                stack="center",
                impute=None,
                title=None,
                axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True),
            ),
            alt.Y("drop:Q", title="Confidence Drop"),
            alt.Color("strategy:N", title="Strategy"),
            tooltip=["strategy", "drop"],
        )
    )
    ticks = (
        alt.Chart(vdf)
        .mark_tick(size=8, opacity=0.5)
        .encode(
            y=alt.Y("drop:Q"),
            color=alt.Color("strategy:N", legend=None),
            tooltip=["strategy", "drop"],
        )
    )
    violin = alt.layer(density, ticks).properties(width=300, height=200)
    # Compose dashboard
    dash = (
        alt.vconcat(
            (bar + eb).resolve_scale(y="independent").properties(height=200),
            heat.properties(height=200),
            violin.properties(height=200),
        )
        .resolve_scale(color="independent")
        .properties(title=f"{model_name} Augmentation Effects ({taxonomy.upper()})")
    )
    return dash


def summarize_augmentation(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute per‐strategy mean preservation rate and average confidence drop.
    """
    # If no data or missing the key column, return empty summary
    if df.height == 0 or "strategy" not in df.columns:
        return pl.DataFrame(
            {
                "strategy": [],
                "preservation_rate": [],
                "avg_conf_drop": [],
            }
        )
    grp = df.group_by("strategy").agg(
        [
            pl.col("preserved").mean().alias("preservation_rate"),
            pl.col("confidence_drop").mean().alias("avg_conf_drop"),
        ]
    )
    return grp


def plot_augmentation_summary(
    df: pl.DataFrame, model_name: str, taxonomy: str
) -> alt.Chart:
    """
    Plot side‐by‐side bars for preservation and avg confidence drop.
    """
    pdf = df.to_pandas()
    base = (
        alt.Chart(pdf)
        .transform_fold(["preservation_rate", "avg_conf_drop"], as_=["metric", "value"])
        .mark_bar()
        .encode(
            x=alt.X("strategy:N", title="Augmentation Strategy"),
            y=alt.Y("value:Q", title="Metric Value"),
            color="metric:N",
            column=alt.Column("metric:N", title=None),
        )
        .properties(title=f"{model_name} – {taxonomy.upper()} Augmentation Effects")
    )
    return base
