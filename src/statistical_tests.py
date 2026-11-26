"""Statistical hypothesis testing and odds ratio modeling."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

from src.config import (
    ADA_GLUCOSE_BINS,
    ADA_GLUCOSE_LABELS,
    ALPHA,
    CLEANED_EDA_PATH,
    FIGURES_DIR,
    WHO_BMI_BINS,
    WHO_BMI_LABELS,
)


def add_clinical_test_bins(df):
    """Add WHO BMI and ADA glucose tiers for chi-square testing."""
    out = df.copy()
    out["bmi_category"] = pd.cut(
        out["bmi"], bins=WHO_BMI_BINS, labels=WHO_BMI_LABELS,
        right=False, include_lowest=True,
    ).astype(str)
    out["glucose_risk_tier"] = pd.cut(
        out["avg_glucose_level"], bins=ADA_GLUCOSE_BINS, labels=ADA_GLUCOSE_LABELS,
        right=False, include_lowest=True,
    ).astype(str)
    return out


def run_chi_square_suite(df, features=None, target="stroke", alpha=ALPHA):
    """Run chi-square tests of independence with Cramer's V effect sizes."""
    binned = add_clinical_test_bins(df)
    if features is None:
        features = [
            "hypertension", "heart_disease", "glucose_risk_tier",
            "bmi_category", "smoking_status", "ever_married",
            "work_type", "Residence_type", "gender", "bmi_was_missing",
        ]

    rows = []
    n_total = len(binned)

    for feat in features:
        ct = pd.crosstab(binned[feat], binned[target])
        chi2, p_val, dof, expected = stats.chi2_contingency(ct)
        min_dim = min(ct.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (n_total * min_dim))) if min_dim > 0 else 0.0
        rows.append({
            "feature": feat,
            "chi2": float(chi2),
            "dof": int(dof),
            "p_value": float(p_val),
            "cramers_v": cramers_v,
            "min_expected_cell": float(expected.min()),
            "decision": "Reject H0 (Significant)" if p_val < alpha else "Fail to Reject H0",
        })

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def run_continuous_group_tests(df, features=None, target="stroke", alpha=ALPHA):
    """Run normality checks, Mann-Whitney U, and Welch's t-test for continuous features."""
    if features is None:
        features = ["age", "avg_glucose_level", "bmi"]

    rows = []
    for feat in features:
        g0 = df.loc[df[target] == 0, feat].dropna()
        g1 = df.loc[df[target] == 1, feat].dropna()

        # normality checks
        _, dagostino_p = stats.normaltest(df[feat].dropna())
        shapiro_sample = df[feat].dropna().sample(n=min(4999, len(df)), random_state=42)
        _, shapiro_p = stats.shapiro(shapiro_sample)

        u_stat, mw_p = stats.mannwhitneyu(g1, g0, alternative="two-sided")
        t_stat, welch_p = stats.ttest_ind(g1, g0, equal_var=False)

        rows.append({
            "feature": feat,
            "non_stroke_median": float(g0.median()),
            "non_stroke_iqr": float(g0.quantile(0.75) - g0.quantile(0.25)),
            "stroke_median": float(g1.median()),
            "stroke_iqr": float(g1.quantile(0.75) - g1.quantile(0.25)),
            "non_stroke_mean": float(g0.mean()),
            "stroke_mean": float(g1.mean()),
            "shapiro_p": float(shapiro_p),
            "dagostino_p": float(dagostino_p),
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p": float(mw_p),
            "welch_t": float(t_stat),
            "welch_p": float(welch_p),
            "decision": "Reject H0 (Significant)" if mw_p < alpha else "Fail to Reject H0",
        })

    return pd.DataFrame(rows)


def fit_epidemiological_logit(df):
    """Fit a multivariable logit model and return odds ratios.

    Continuous predictors are scaled to clinically useful increments:
    age per 10 years, glucose per 10 mg/dL, BMI per 5 units.
    """
    model_df = pd.DataFrame(index=df.index)
    model_df["age_per_10yr"] = df["age"] / 10.0
    model_df["glucose_per_10mgdl"] = df["avg_glucose_level"] / 10.0
    model_df["bmi_per_5unit"] = df["bmi"] / 5.0
    model_df["hypertension"] = df["hypertension"].astype(float)
    model_df["heart_disease"] = df["heart_disease"].astype(float)
    model_df["bmi_was_missing"] = df["bmi_was_missing"].astype(float)
    model_df["male_gender"] = (df["gender"] == "Male").astype(float)
    model_df["ever_married"] = (df["ever_married"] == "Yes").astype(float)
    model_df["urban_residence"] = (df["Residence_type"] == "Urban").astype(float)

    # reference level for smoking: 'never smoked'
    for status in ["formerly smoked", "smokes", "Unknown"]:
        col_name = f"smoking_{status.replace(' ', '_')}"
        model_df[col_name] = (df["smoking_status"] == status).astype(float)

    X = sm.add_constant(model_df, has_constant="add")
    y = df["stroke"].astype(float)

    logit_res = sm.Logit(y, X).fit(disp=False, maxiter=200)

    ci = logit_res.conf_int()
    or_df = pd.DataFrame({
        "predictor": logit_res.params.index,
        "beta": logit_res.params.values,
        "std_err": logit_res.bse.values,
        "odds_ratio": np.exp(logit_res.params.values),
        "ci_lower_95": np.exp(ci[0].values),
        "ci_upper_95": np.exp(ci[1].values),
        "p_value": logit_res.pvalues.values,
    })
    or_df["significant"] = or_df["p_value"] < ALPHA
    or_df = or_df[or_df["predictor"] != "const"].reset_index(drop=True)
    return logit_res, or_df


def plot_odds_ratios_forest(or_df, save_path=FIGURES_DIR / "07_odds_ratios_forest_plot.png"):
    """Create a forest plot of adjusted odds ratios with 95% CIs."""
    plot_df = or_df.sort_values("odds_ratio", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    colors = ["#c53030" if sig else "#718096" for sig in plot_df["significant"]]

    for idx, row in plot_df.iterrows():
        ax.errorbar(
            x=row["odds_ratio"], y=idx,
            xerr=[[row["odds_ratio"] - row["ci_lower_95"]], [row["ci_upper_95"] - row["odds_ratio"]]],
            fmt="o", color=colors[idx], ecolor=colors[idx],
            elinewidth=2.0, capsize=4, markersize=7,
        )
        ax.text(
            row["ci_upper_95"] * 1.04, idx,
            f"OR={row['odds_ratio']:.2f} (p={row['p_value']:.3g})",
            va="center", fontsize=8.8, color="#1a202c",
        )

    ax.axvline(1.0, color="#1a202c", linestyle="--", linewidth=1.2, label="Null Effect (OR = 1.0)")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["predictor"])
    ax.set_xscale("log")
    ax.set_xlim(0.35, max(plot_df["ci_upper_95"].max() * 2.2, 12.0))
    ax.set_xlabel("Adjusted Odds Ratio (Log Scale, 95% CI)")
    ax.set_title("Figure 7: Multivariable Logistic Regression Adjusted Odds Ratios (95% CI)",
                 fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    sns.despine(ax=ax)
    plt.tight_layout()

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_all_statistical_tests(eda_csv_path=CLEANED_EDA_PATH, figures_dir=FIGURES_DIR):
    """Run the full stats suite: chi-square, continuous tests, logit, and forest plot."""
    df = pd.read_csv(eda_csv_path)
    chi_df = run_chi_square_suite(df)
    cont_df = run_continuous_group_tests(df)
    logit_res, or_df = fit_epidemiological_logit(df)

    fig_dir = Path(figures_dir)
    forest_path = plot_odds_ratios_forest(or_df, fig_dir / "07_odds_ratios_forest_plot.png")

    # save everything to JSON for the app
    summary_payload = {
        "n_observations": int(len(df)),
        "alpha": ALPHA,
        "logit_pseudo_r2": float(logit_res.prsquared),
        "logit_llr_pvalue": float(logit_res.llr_pvalue),
        "logit_aic": float(logit_res.aic),
        "chi_square_results": chi_df.to_dict(orient="records"),
        "continuous_group_tests": cont_df.to_dict(orient="records"),
        "logit_odds_ratios": or_df.to_dict(orient="records"),
    }
    json_path = fig_dir / "statistical_summary.json"
    json_path.write_text(json.dumps(summary_payload, indent=2))

    return {
        "chi_square": chi_df,
        "continuous": cont_df,
        "odds_ratios": or_df,
        "logit_summary": logit_res,
        "forest_plot_path": str(forest_path),
        "summary_json_path": str(json_path),
    }


if __name__ == "__main__":
    results = run_all_statistical_tests()
    print("=== CHI-SQUARE TESTS ===")
    print(results["chi_square"][["feature", "chi2", "dof", "p_value", "cramers_v", "decision"]].to_string(index=False))
    print("\n=== CONTINUOUS GROUP TESTS ===")
    print(results["continuous"][["feature", "non_stroke_median", "stroke_median", "mann_whitney_p", "welch_p", "decision"]].to_string(index=False))
    print("\n=== LOGIT ADJUSTED ODDS RATIOS ===")
    print(results["odds_ratios"][["predictor", "odds_ratio", "ci_lower_95", "ci_upper_95", "p_value", "significant"]].to_string(index=False))
    print("\nSaved forest plot:", results["forest_plot_path"])
    print("Saved JSON tables:", results["summary_json_path"])
