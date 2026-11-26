"""Model training, cross-validation, threshold calibration, and saving results."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.config import (
    BEST_MODEL_PATH,
    CV_FOLDS,
    DATA_RAW_PATH,
    FIGURES_DIR,
    LR_MODEL_PATH,
    METRICS_PATH,
    MODELS_DIR,
    RANDOM_STATE,
)
from src.feature_engineering import load_stratified_train_test, make_modeling_pipeline


def compute_classification_metrics(y_true, y_prob, threshold=0.50):
    """Compute classification metrics at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def calibrate_triage_thresholds(y_true, oof_probs, target_recall=0.80):
    """Find screening threshold (>= target recall) and Youden's J threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, oof_probs)
    valid_mask = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[valid_mask], tpr[valid_mask], thresholds[valid_mask]

    # Youden's J = TPR - FPR (maximizes the tradeoff)
    j_scores = tpr - fpr
    youden_idx = int(np.argmax(j_scores))
    youden_thresh = float(thresholds[youden_idx])

    # find the highest threshold that still gives >= target recall
    recall_candidates = np.where(tpr >= target_recall)[0]
    if len(recall_candidates) > 0:
        screen_thresh = float(thresholds[recall_candidates[0]])
    else:
        screen_thresh = youden_thresh

    # set up 3-tier triage bands
    low_thresh = float(np.clip(round(screen_thresh, 2), 0.20, 0.55))
    high_thresh = float(np.clip(round(max(youden_thresh + 0.15, low_thresh + 0.20), 2), 0.45, 0.80))
    if high_thresh <= low_thresh + 0.10:
        high_thresh = round(min(low_thresh + 0.20, 0.85), 2)

    return {
        "screen_threshold": round(screen_thresh, 4),
        "youden_threshold": round(youden_thresh, 4),
        "low_risk_max": low_thresh,
        "high_risk_min": high_thresh,
    }


def build_threshold_sweep(y_true, y_prob):
    """Compute metrics across thresholds 0.10 to 0.90 for the threshold lab."""
    rows = []
    for t in np.round(np.arange(0.10, 0.91, 0.05), 2):
        rows.append(compute_classification_metrics(y_true, y_prob, threshold=float(t)))
    return rows


def extract_feature_importances(lr_pipe, ensemble_pipe, top_n=12):
    """Get top feature importances from LR coefficients and ensemble importances."""
    feat_names = list(lr_pipe.named_steps["preprocessor"].get_feature_names_out())
    lr_coefs = lr_pipe.named_steps["classifier"].coef_[0]
    lr_df = pd.DataFrame({
        "feature": feat_names,
        "coefficient": lr_coefs,
        "abs_importance": np.abs(lr_coefs),
    }).sort_values("abs_importance", ascending=False)

    ens_names = list(ensemble_pipe.named_steps["preprocessor"].get_feature_names_out())
    ens_imp = ensemble_pipe.named_steps["classifier"].feature_importances_
    ens_df = pd.DataFrame({
        "feature": ens_names,
        "importance": ens_imp,
    }).sort_values("importance", ascending=False)

    return {
        "logistic_regression": lr_df.head(top_n).to_dict(orient="records"),
        "best_ensemble": ens_df.head(top_n).to_dict(orient="records"),
    }


def _save_roc_pr_plot(y_test, test_probs_by_model, save_path):
    """Save combined ROC and PR curve comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    palette = ["#c53030", "#2b6cb0", "#2f855a", "#d69e2e", "#805ad5"]

    for (name, probs), color in zip(test_probs_by_model.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_val = roc_auc_score(y_test, probs)
        axes[0].plot(fpr, tpr, lw=2.0, color=color, label=f"{name} (AUC = {roc_val:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_val = average_precision_score(y_test, probs)
        axes[1].plot(rec, prec, lw=2.0, color=color, label=f"{name} (AP = {pr_val:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1.1, label="Random Baseline (0.500)")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)")
    axes[0].set_title("ROC Curve Comparison (Test Set, n=1,022)", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8.5)
    sns.despine(ax=axes[0])

    baseline = float(y_test.mean())
    axes[1].axhline(baseline, color="black", linestyle="--", lw=1.1, label=f"No-Skill Baseline ({baseline:.3f})")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (PPV)")
    axes[1].set_title("Precision-Recall Curve Comparison (Imbalance-Aware)", fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8.5)
    sns.despine(ax=axes[1])

    fig.suptitle("Figure 8: Multi-Model Discrimination & Precision-Recall Benchmarks", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_confusion_matrices_plot(y_test, lr_probs, ens_probs, lr_thresh,
                                   ens_thresh, ens_name, save_path):
    """Save confusion matrices at the calibrated thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    specs = [
        (axes[0], lr_probs, lr_thresh, f"Logistic Regression (Thresh = {lr_thresh:.2f})"),
        (axes[1], ens_probs, ens_thresh, f"{ens_name} (Thresh = {ens_thresh:.2f})"),
    ]
    labels = ["No Stroke (0)", "Stroke (1)"]

    for ax, probs, thresh, title in specs:
        preds = (probs >= thresh).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 12, "weight": "bold"}, ax=ax,
        )
        rec = recall_score(y_test, preds, zero_division=0)
        spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        ax.set_xlabel("Predicted Clinical Triage")
        ax.set_ylabel("Actual Outcome")
        ax.set_title(f"{title}\nRecall={rec:.1%} | Specificity={spec:.1%}", fontweight="bold")

    fig.suptitle("Figure 9: Test-Set Confusion Matrices at Calibrated Screening Thresholds", fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_feature_importance_plot(importances, ens_name, save_path):
    """Save feature importance comparison: LR coefficients vs ensemble importances."""
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8))

    lr_df = pd.DataFrame(importances["logistic_regression"]).iloc[::-1]
    lr_colors = ["#c53030" if c > 0 else "#2b6cb0" for c in lr_df["coefficient"]]
    axes[0].barh(lr_df["feature"], lr_df["coefficient"], color=lr_colors, edgecolor="#1a202c")
    axes[0].axvline(0, color="#1a202c", linewidth=1.0)
    axes[0].set_xlabel("Standardized Log-Odds Coefficient (β)")
    axes[0].set_title("Logistic Regression Top Risk Drivers\n(Red = Increases Risk, Blue = Decreases Risk)", fontweight="bold")
    sns.despine(ax=axes[0])

    ens_df = pd.DataFrame(importances["best_ensemble"]).iloc[::-1]
    axes[1].barh(ens_df["feature"], ens_df["importance"], color="#2f855a", edgecolor="#1a202c")
    axes[1].set_xlabel("Normalized Feature Importance")
    axes[1].set_title(f"{ens_name} Top Feature Importances", fontweight="bold")
    sns.despine(ax=axes[1])

    fig.suptitle("Figure 10: Clinical Feature Importance — Interpretable Linear vs. Ensemble", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate_all(raw_path=DATA_RAW_PATH, models_dir=MODELS_DIR,
                            figures_dir=FIGURES_DIR):
    """Train all 5 models, evaluate, calibrate thresholds, save plots and models."""
    models_path = Path(models_dir)
    figs_path = Path(figures_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    figs_path.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_stratified_train_test(raw_path)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    neg_pos_ratio = float((y_train == 0).sum() / (y_train == 1).sum())

    # --- imbalance strategy comparison ---
    imbalance_comparison = []
    for name, est_factory, smote_flag in [
        ("Logistic Regression (class_weight='balanced')",
         lambda: LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE), False),
        ("Logistic Regression (SMOTE in CV)",
         lambda: LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE), True),
        ("Random Forest (class_weight='balanced')",
         lambda: RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=8,
                                        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), False),
        ("Random Forest (SMOTE in CV)",
         lambda: RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=8,
                                        random_state=RANDOM_STATE, n_jobs=-1), True),
    ]:
        pipe = make_modeling_pipeline(est_factory(), use_smote=smote_flag)
        oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        imbalance_comparison.append({
            "experiment": name,
            "cv_roc_auc": float(roc_auc_score(y_train, oof)),
            "cv_pr_auc": float(average_precision_score(y_train, oof)),
            "cv_f1_at_050": float(f1_score(y_train, (oof >= 0.5).astype(int))),
            "cv_recall_at_050": float(recall_score(y_train, (oof >= 0.5).astype(int))),
        })

    # --- benchmark 5 model families with GridSearchCV ---
    candidates = {
        "Logistic Regression": (
            make_modeling_pipeline(
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
                use_smote=False,
            ),
            {"classifier__C": [0.05, 0.1, 0.5, 1.0, 2.0]},
        ),
        "Random Forest": (
            make_modeling_pipeline(
                RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
                use_smote=False,
            ),
            {
                "classifier__n_estimators": [250, 400],
                "classifier__max_depth": [4, 6, 8],
                "classifier__min_samples_leaf": [5, 10],
            },
        ),
        "XGBoost": (
            make_modeling_pipeline(
                XGBClassifier(
                    scale_pos_weight=neg_pos_ratio, eval_metric="logloss",
                    random_state=RANDOM_STATE, n_jobs=-1,
                ),
                use_smote=False,
            ),
            {
                "classifier__n_estimators": [150, 250],
                "classifier__max_depth": [2, 3],
                "classifier__learning_rate": [0.03, 0.05],
                "classifier__subsample": [0.8],
            },
        ),
        "K-Nearest Neighbors (SMOTE)": (
            make_modeling_pipeline(KNeighborsClassifier(), use_smote=True),
            {"classifier__n_neighbors": [15, 25, 35], "classifier__weights": ["uniform", "distance"]},
        ),
        "Gaussian Naive Bayes": (
            make_modeling_pipeline(GaussianNB(), use_smote=False),
            {"classifier__var_smoothing": [1e-9, 1e-6, 1e-3]},
        ),
    }

    fitted_models = {}
    oof_probs_by_model = {}
    test_probs_by_model = {}
    leaderboard = []
    calibrations = {}

    for model_name, (pipe, param_grid) in candidates.items():
        gs = GridSearchCV(
            estimator=pipe, param_grid=param_grid, cv=cv,
            scoring="roc_auc", n_jobs=-1, refit=True,
        )
        gs.fit(X_train, y_train)
        best_pipe = gs.best_estimator_
        fitted_models[model_name] = best_pipe

        # get out-of-fold and test predictions
        oof_probs = cross_val_predict(best_pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        test_probs = best_pipe.predict_proba(X_test)[:, 1]
        oof_probs_by_model[model_name] = oof_probs
        test_probs_by_model[model_name] = test_probs

        # calibrate triage thresholds on OOF predictions
        calib = calibrate_triage_thresholds(y_train, oof_probs, target_recall=0.80)
        calibrations[model_name] = calib

        test_default = compute_classification_metrics(y_test, test_probs, threshold=0.50)
        test_calibrated = compute_classification_metrics(y_test, test_probs, threshold=calib["low_risk_max"])

        leaderboard.append({
            "model": model_name,
            "best_params": {k.replace("classifier__", ""): v for k, v in gs.best_params_.items()},
            "cv_roc_auc": float(roc_auc_score(y_train, oof_probs)),
            "cv_pr_auc": float(average_precision_score(y_train, oof_probs)),
            "calibrated_screen_thresh": calib["low_risk_max"],
            "high_risk_thresh": calib["high_risk_min"],
            "test_roc_auc": test_default["roc_auc"],
            "test_pr_auc": test_default["pr_auc"],
            "test_default_050": test_default,
            "test_calibrated": test_calibrated,
        })

    # pick best ensemble (RF vs XGBoost)
    ensemble_candidates = [row for row in leaderboard if row["model"] in {"Random Forest", "XGBoost"}]
    best_ensemble_row = max(ensemble_candidates, key=lambda r: (r["cv_roc_auc"], r["cv_pr_auc"]))
    best_ensemble_name = best_ensemble_row["model"]
    best_overall_row = max(leaderboard, key=lambda r: (r["cv_roc_auc"], r["cv_pr_auc"]))

    lr_pipe = fitted_models["Logistic Regression"]
    best_ens_pipe = fitted_models[best_ensemble_name]

    # save the trained pipelines
    joblib.dump(lr_pipe, models_path / LR_MODEL_PATH.name)
    joblib.dump(best_ens_pipe, models_path / BEST_MODEL_PATH.name)

    # generate evaluation plots
    _save_roc_pr_plot(y_test, test_probs_by_model, figs_path / "08_model_comparison_roc_pr.png")
    _save_confusion_matrices_plot(
        y_test,
        test_probs_by_model["Logistic Regression"],
        test_probs_by_model[best_ensemble_name],
        calibrations["Logistic Regression"]["low_risk_max"],
        calibrations[best_ensemble_name]["low_risk_max"],
        best_ensemble_name,
        figs_path / "09_confusion_matrix_best_model.png",
    )
    feat_importances = extract_feature_importances(lr_pipe, best_ens_pipe, top_n=12)
    _save_feature_importance_plot(feat_importances, best_ensemble_name, figs_path / "10_feature_importance.png")

    # error analysis on the LR model
    lr_thresh = calibrations["Logistic Regression"]["low_risk_max"]
    lr_test_probs = test_probs_by_model["Logistic Regression"]
    lr_preds = (lr_test_probs >= lr_thresh).astype(int)
    err_df = X_test.copy()
    err_df["actual_stroke"] = y_test.values
    err_df["pred_prob"] = lr_test_probs
    err_df["pred_label"] = lr_preds

    fn_df = err_df[(err_df["actual_stroke"] == 1) & (err_df["pred_label"] == 0)]
    fp_df = err_df[(err_df["actual_stroke"] == 0) & (err_df["pred_label"] == 1)]
    tp_df = err_df[(err_df["actual_stroke"] == 1) & (err_df["pred_label"] == 1)]

    error_analysis = {
        "model": "Logistic Regression",
        "threshold": lr_thresh,
        "true_positives_count": int(len(tp_df)),
        "false_negatives_count": int(len(fn_df)),
        "false_positives_count": int(len(fp_df)),
        "false_negative_median_age": float(fn_df["age"].median()) if not fn_df.empty else None,
        "true_positive_median_age": float(tp_df["age"].median()) if not tp_df.empty else None,
        "false_negative_median_glucose": float(fn_df["avg_glucose_level"].median()) if not fn_df.empty else None,
        "false_positive_median_age": float(fp_df["age"].median()) if not fp_df.empty else None,
        "false_negative_samples": fn_df[
            ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "smoking_status", "pred_prob"]
        ].head(5).to_dict(orient="records"),
    }

    # precompute ROC/PR curve data for the streamlit app
    curve_data = {}
    for name, probs in test_probs_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        prec, rec, _ = precision_recall_curve(y_test, probs)
        step = max(1, len(rec) // 120)
        curve_data[name] = {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr],
            "pr_recall": [round(float(x), 4) for x in rec[::step]],
            "pr_precision": [round(float(x), 4) for x in prec[::step]],
        }

    # save everything to JSON
    metrics_payload = {
        "dataset_split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_stroke_rate": float(y_train.mean()),
            "test_stroke_rate": float(y_test.mean()),
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
        },
        "best_overall_model": best_overall_row["model"],
        "best_ensemble_model": best_ensemble_name,
        "triage_thresholds": {
            "Logistic Regression": calibrations["Logistic Regression"],
            best_ensemble_name: calibrations[best_ensemble_name],
        },
        "imbalance_strategy_comparison": imbalance_comparison,
        "leaderboard": leaderboard,
        "threshold_sweeps": {
            "Logistic Regression": build_threshold_sweep(y_test, test_probs_by_model["Logistic Regression"]),
            best_ensemble_name: build_threshold_sweep(y_test, test_probs_by_model[best_ensemble_name]),
        },
        "feature_importances": feat_importances,
        "error_analysis": error_analysis,
        "curve_data": curve_data,
    }

    (models_path / METRICS_PATH.name).write_text(json.dumps(metrics_payload, indent=2))
    return metrics_payload


if __name__ == "__main__":
    payload = train_and_evaluate_all()
    print("=== IMBALANCE STRATEGY COMPARISON ===")
    print(pd.DataFrame(payload["imbalance_strategy_comparison"]).round(4).to_string(index=False))
    print("\n=== 5-MODEL BENCHMARK LEADERBOARD ===")
    summary_rows = []
    for row in payload["leaderboard"]:
        cal = row["test_calibrated"]
        summary_rows.append({
            "Model": row["model"],
            "CV_ROC": round(row["cv_roc_auc"], 4),
            "CV_PR": round(row["cv_pr_auc"], 4),
            "Test_ROC": round(row["test_roc_auc"], 4),
            "Test_PR": round(row["test_pr_auc"], 4),
            "Thresh": row["calibrated_screen_thresh"],
            "Recall": round(cal["recall_sensitivity"], 4),
            "Spec": round(cal["specificity"], 4),
            "F1": round(cal["f1_score"], 4),
        })
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nBest overall CV model:", payload["best_overall_model"])
    print("Best ensemble model:", payload["best_ensemble_model"])
