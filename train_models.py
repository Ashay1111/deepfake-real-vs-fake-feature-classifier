from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Keep matplotlib cache writable inside workspace to avoid font/cache warnings.
os.environ.setdefault("MPLCONFIGDIR", str((Path(".") / ".mplconfig").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- config --------------------------------------------------------------------
OUT = Path("results")
OUT.mkdir(exist_ok=True)
FEATS = [
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
    "mean_patch_entropy",
    "std_patch_entropy",
    "high_entropy_patch_ratio",
    "num_clusters",
]

# -- load ----------------------------------------------------------------------
train = pd.read_csv("features/train_selected.csv")
val = pd.read_csv("features/val_selected.csv")

X_train = train[FEATS].values
y_train = (train["label"] == "fake").astype(int).values

X_val = val[FEATS].values
y_val = (val["label"] == "fake").astype(int).values

# -- scale (required for logistic regression) ----------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

scaler_params = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "features": FEATS,
}
(OUT / "scaler_params.json").write_text(json.dumps(scaler_params, indent=2))


# -- evaluation helper ----------------------------------------------------------
def evaluate(name, model, X, y, df_meta, inputs_are_scaled=False):
    X_eval = X if inputs_are_scaled else scaler.transform(X)
    preds = model.predict(X_eval)
    proba = model.predict_proba(X_eval)[:, 1]

    print(f'\n{"─"*55}')
    print(f"  {name}")
    print(f'{"─"*55}')
    print(classification_report(y, preds, target_names=["real", "fake"]))
    print(f"  ROC-AUC : {roc_auc_score(y, proba):.4f}")
    print(f"  F1(fake): {f1_score(y, preds):.4f}")

    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["real", "fake"],
        yticklabels=["real", "fake"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} - confusion matrix (val)")
    plt.tight_layout()
    plt.savefig(OUT / f'cm_{name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()

    print("\n  Per-method F1 (val):")
    df_meta = df_meta.copy()
    df_meta["pred"] = preds
    df_meta["proba"] = proba
    df_meta["true"] = y
    for method in sorted(df_meta["method"].unique()):
        sub = df_meta[df_meta["method"] == method]
        f1 = f1_score(sub["true"], sub["pred"], zero_division=0)
        auc = (
            roc_auc_score(sub["true"], sub["proba"])
            if sub["true"].nunique() > 1
            else float("nan")
        )
        n = len(sub)
        print(f"    {method:15s}  F1={f1:.3f}  AUC={auc:.3f}  n={n}")

    return {
        "name": name,
        "f1": f1_score(y, preds),
        "auc": roc_auc_score(y, proba),
        "preds": preds,
        "proba": proba,
    }


# -- 1. logistic regression ----------------------------------------------------
lr = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)
lr.fit(X_train_s, y_train)

lr_results = evaluate(
    "Logistic Regression", lr, X_val_s, y_val, val, inputs_are_scaled=True
)

coef_df = pd.DataFrame({"feature": FEATS, "coefficient": lr.coef_[0]}).sort_values(
    "coefficient", ascending=False
)
coef_df.to_csv(OUT / "lr_coefficients.csv", index=False)
print("\n  LR feature coefficients (top positive -> fake, top negative -> real):")
print(coef_df.to_string(index=False))


# -- 2. random forest ----------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

rf_results = evaluate("Random Forest", rf, X_val, y_val, val, inputs_are_scaled=True)

imp_df = pd.DataFrame({"feature": FEATS, "importance": rf.feature_importances_}).sort_values(
    "importance", ascending=False
)
imp_df.to_csv(OUT / "rf_importances.csv", index=False)
print("\n  RF feature importances:")
print(imp_df.to_string(index=False))


# -- 3. gradient boosting ------------------------------------------------------
gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42,
)
gb.fit(X_train, y_train)

gb_results = evaluate(
    "Gradient Boosting", gb, X_val, y_val, val, inputs_are_scaled=True
)

imp_gb = pd.DataFrame({"feature": FEATS, "importance": gb.feature_importances_}).sort_values(
    "importance", ascending=False
)
imp_gb.to_csv(OUT / "gb_importances.csv", index=False)


# -- summary -------------------------------------------------------------------
print("\n\n" + "═" * 55)
print("  MODEL COMPARISON (val set)")
print("═" * 55)
for r in [lr_results, rf_results, gb_results]:
    print(f"  {r['name']:25s}  F1={r['f1']:.4f}  AUC={r['auc']:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (title, df) in zip(
    axes,
    [
        (
            "LR |coefficient|",
            coef_df.assign(importance=coef_df["coefficient"].abs()).sort_values(
                "importance", ascending=True
            ),
        ),
        ("Random Forest", imp_df.sort_values("importance", ascending=True)),
        ("Gradient Boosting", imp_gb.sort_values("importance", ascending=True)),
    ],
):
    col = "importance"
    ax.barh(df["feature"], df[col], color="#7c6af7", alpha=0.8)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Feature importance across models", fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "feature_importances_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nsaved: results/feature_importances_comparison.png")
