from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Keep matplotlib cache in workspace.
os.environ.setdefault("MPLCONFIGDIR", str((Path(".") / ".mplconfig").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT = Path("results")
OUT.mkdir(exist_ok=True)

FEATURES_FINAL = [
    "mean_intensity",
    "std_intensity",
    "residual_variance",
    "autocorr_strength",
    "offcenter_energy",
    "near_center_energy",
    "radial_loglog_slope",
    "radial_loglog_intercept",
    "high_band_power",
    "low_high_ratio",
]

# -- load all three splits -----------------------------------------------------
train = pd.read_csv("features/train_selected.csv")
val = pd.read_csv("features/val_selected.csv")
test = pd.read_csv("features/test_selected.csv")

# train on train+val for final model
train_full = pd.concat([train, val], ignore_index=True)

X_train = train_full[FEATURES_FINAL].values
y_train = (train_full["label"] == "fake").astype(int).values

X_test = test[FEATURES_FINAL].values
y_test = (test["label"] == "fake").astype(int).values

# -- train final model ---------------------------------------------------------
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

# -- overall metrics -----------------------------------------------------------
print("=" * 60)
print("  FINAL TEST SET RESULTS")
print("=" * 60)
print(classification_report(y_test, preds, target_names=["real", "fake"]))
print(f"  ROC-AUC  : {roc_auc_score(y_test, proba):.4f}")
print(f"  F1       : {f1_score(y_test, preds):.4f}")
print(f"  Precision: {precision_score(y_test, preds):.4f}")
print(f"  Recall   : {recall_score(y_test, preds):.4f}")

# -- per-method breakdown ------------------------------------------------------
print("\n  Per-method breakdown:")
print(f'  {"method":15s}  {"F1":>6}  {"AUC":>6}  {"Prec":>6}  {"Rec":>6}  {"n":>5}')
print("  " + "─" * 50)

method_results = []
for method in sorted(test["method"].unique()):
    mask = (test["method"] == method).values
    yt = y_test[mask]
    yp = preds[mask]
    ypr = proba[mask]

    f1 = f1_score(yt, yp, zero_division=0)
    auc = roc_auc_score(yt, ypr) if pd.Series(yt).nunique() > 1 else float("nan")
    prec = precision_score(yt, yp, zero_division=0)
    rec = recall_score(yt, yp, zero_division=0)
    n = int(mask.sum())

    method_results.append(
        {
            "method": method,
            "f1": f1,
            "auc": auc,
            "precision": prec,
            "recall": rec,
            "n": n,
        }
    )
    print(f"  {method:15s}  {f1:6.4f}  {auc:6.4f}  {prec:6.4f}  {rec:6.4f}  {n:5d}")

# -- confusion matrix ----------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(22, 4))

cm = confusion_matrix(y_test, preds)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["real", "fake"],
    yticklabels=["real", "fake"],
    ax=axes[0],
)
axes[0].set_title("Overall", fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

for ax, res in zip(axes[1:], method_results):
    mask = (test["method"] == res["method"]).values
    cm_m = confusion_matrix(y_test[mask], preds[mask])
    sns.heatmap(
        cm_m,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["real", "fake"],
        yticklabels=["real", "fake"],
        ax=ax,
    )
    ax.set_title(f"{res['method']}\nF1={res['f1']:.3f}", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("")

plt.suptitle("Final model - confusion matrices (test set)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "final_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nsaved: results/final_confusion_matrices.png")

# -- ROC curves ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#7c6af7", "#4ecdc4", "#f7847a", "#f7c56a"]

for res, color in zip(method_results, colors):
    mask = (test["method"] == res["method"]).values
    fpr, tpr, _ = roc_curve(y_test[mask], proba[mask])
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{res['method']} (AUC={res['auc']:.3f})")

fpr, tpr, _ = roc_curve(y_test, proba)
ax.plot(
    fpr,
    tpr,
    color="white",
    lw=2.5,
    linestyle="--",
    label=f"Overall (AUC={roc_auc_score(y_test, proba):.3f})",
)
ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves by method - test set", fontweight="bold")
ax.legend(loc="lower right")
ax.set_facecolor("#0a0a0f")
fig.patch.set_facecolor("#0a0a0f")
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")
ax.legend(loc="lower right", facecolor="#1a1a26", edgecolor="gray", labelcolor="white")
plt.tight_layout()
plt.savefig(OUT / "final_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved: results/final_roc_curves.png")

# -- save summary --------------------------------------------------------------
summary = {
    "model": "GradientBoostingClassifier",
    "features": FEATURES_FINAL,
    "n_features": len(FEATURES_FINAL),
    "test_f1": round(f1_score(y_test, preds), 4),
    "test_auc": round(roc_auc_score(y_test, proba), 4),
    "test_precision": round(precision_score(y_test, preds), 4),
    "test_recall": round(recall_score(y_test, preds), 4),
    "per_method": method_results,
}
(OUT / "final_results.json").write_text(json.dumps(summary, indent=2))
print("saved: results/final_results.json")
