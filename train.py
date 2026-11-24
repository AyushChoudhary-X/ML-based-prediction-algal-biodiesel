"""Run once:  python train.py   -> writes everything the Streamlit app displays into ./artifacts"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import core
from core import FEATURES, NUM, CAT, TARGET, NICE

warnings.filterwarnings("ignore")
ART = core.ROOT / "artifacts"; FIG = ART / "figs"; FIG.mkdir(parents=True, exist_ok=True)
plt.style.use("dark_background")
BG = "#0e1117"
PAL = ["#8ab4f8", "#4f7cf0", "#f4a6a6", "#ef5350", "#98e6a8", "#4db6a0"]

d = core.load_data()
X, y, yc, grp = d[FEATURES], d[TARGET], d["High_Yield"], d["Study_ID"]


def fit_pred(name, task, tr, te):
    m = core.make_model(name, task).fit(X.iloc[tr], (y if task == "reg" else yc).iloc[tr])
    return m.predict(X.iloc[te])


def random_cv(name, task, n_rep=3):
    yy = y if task == "reg" else yc
    pred = np.zeros((n_rep, len(d)))
    splitter = (RepeatedKFold(n_splits=5, n_repeats=n_rep, random_state=0) if task == "reg"
                else RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=0))
    for k, (tr, te) in enumerate(splitter.split(X, yc)):
        pred[k // 5, te] = fit_pred(name, task, tr, te)
    return pred


# ------------------------------------------------------------------ regression benchmark
reg_rows, reg_cv = [], {}
for name in core.REG:
    pr = random_cv(name, "reg"); reg_cv[name] = pr
    reg_rows.append({"Model": name, "R2": np.mean([r2_score(y, p) for p in pr]),
                     "RMSE": np.mean([np.sqrt(mean_squared_error(y, p)) for p in pr]),
                     "MAE": np.mean([mean_absolute_error(y, p) for p in pr])})
reg = pd.DataFrame(reg_rows).sort_values("R2", ascending=False).reset_index(drop=True)
reg.loc[len(reg)] = ["Baseline (predict mean)", 0.0, y.std(), np.mean(np.abs(y - y.mean()))]
reg.to_csv(ART / "metrics_regression.csv", index=False)

# ------------------------------------------------------------------ classification benchmark (High vs Low yield)
cls_rows = []
for name in core.CLS:
    pr = random_cv(name, "cls")
    cls_rows.append({"Model": name, "Accuracy": np.mean([accuracy_score(yc, p) for p in pr]),
                     "F1": np.mean([f1_score(yc, p, average="weighted") for p in pr])})
cls = pd.DataFrame(cls_rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
cls.loc[len(cls)] = ["Baseline (majority class)", max(yc.mean(), 1 - yc.mean()), np.nan]
cls.to_csv(ART / "metrics_classification.csv", index=False)

best_reg = reg.iloc[0]["Model"]
print(reg.round(3).to_string()); print(cls.round(3).to_string())

# ------------------------------------------------------------------ importance
models = core.fit_final(d)
blend = core.BlendModel(models)
gb = models["gb"]
names = gb[:-1].get_feature_names_out()
imp_tree = pd.Series(gb["m"].feature_importances_, index=[n.split("__", 1)[1] for n in names]).sort_values(ascending=False)
imp_tree.to_csv(ART / "importance_tree.csv", header=["importance"])


def perm_importance(model_name, splits, n_rep=5):
    rng = np.random.default_rng(0); out = {c: [] for c in FEATURES}
    for tr, te in splits:
        m = core.make_model(model_name).fit(X.iloc[tr], y.iloc[tr])
        base_err = mean_squared_error(y.iloc[te], m.predict(X.iloc[te]))
        for c in FEATURES:
            e = []
            for _ in range(n_rep):
                Xp = X.iloc[te].copy(); Xp[c] = rng.permutation(Xp[c].values)
                e.append(mean_squared_error(y.iloc[te], m.predict(Xp)) - base_err)
            out[c].append(np.mean(e))
    return pd.Series({c: np.mean(v) for c, v in out.items()})


rk = list(RepeatedKFold(n_splits=5, n_repeats=1, random_state=1).split(X))
perm = pd.DataFrame({"MSE increase": perm_importance("Gradient Boosting", rk)})
perm.index = [NICE[i] for i in perm.index]
perm.sort_values("MSE increase", ascending=False).to_csv(ART / "importance_permutation.csv")

# ------------------------------------------------------------------ Shapley-style attribution (blended final model)
base_val, sv = core.shapley_mc(blend, X, n_perm=80, seed=0)
shap_df = pd.DataFrame(sv, columns=FEATURES); shap_df.insert(0, "Study_ID", grp.values)
shap_df.to_csv(ART / "shap_values.csv", index=False)

# ------------------------------------------------------------------ partial dependence
pdp = {}
for c in NUM[1:]:
    g, v = core.partial_dependence(blend, X, c); pdp[c] = (g, v)

# ------------------------------------------------------------------ figures
def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight", facecolor=BG); plt.close(fig)


def bar_pair(df, cols, titles, fname, ylabel):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=BG)
    for a, c, t in zip(ax, cols, titles):
        a.set_facecolor(BG); b = a.bar(df["Model"], df[c], color=PAL + ["#888"] * 3)
        a.set_title(t, loc="left", fontweight="bold"); a.set_ylabel(ylabel)
        a.tick_params(axis="x", rotation=40); a.set_ylim(0, 1.05)
        for r_, v in zip(b, df[c]):
            if pd.notna(v): a.text(r_.get_x() + r_.get_width() / 2, v + .01, f"{v:.3f}", ha="center", fontsize=8)
        for s in ("top", "right"): a.spines[s].set_visible(False)
    fig.tight_layout(); save(fig, fname)


c2 = cls[~cls.Model.str.startswith("Baseline")]
bar_pair(c2, ["Accuracy", "F1"], ["Accuracy: random 5-fold CV", "F1-score: random 5-fold CV"], "cls_random.png", "score")

r2 = reg[~reg.Model.str.startswith("Baseline")]
fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=BG); ax.set_facecolor(BG)
b_ = ax.bar(r2.Model, r2["R2"], color=PAL)
for r_, v in zip(b_, r2["R2"]): ax.text(r_.get_x() + r_.get_width() / 2, v + .01, f"{v:.3f}", ha="center", fontsize=8)
ax.set_ylabel("R²"); ax.tick_params(axis="x", rotation=30); ax.set_title("Regression R²: random 5-fold CV", loc="left", fontweight="bold")
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
save(fig, "reg_r2.png")

# parity plot
fig, ax = plt.subplots(figsize=(5.6, 4.8), facecolor=BG); ax.set_facecolor(BG)
cols = dict(zip(sorted(grp.unique()), ["#8ab4f8", "#ffd54f", "#ef5350", "#98e6a8", "#f4a6a6", "#b39ddb", "#4db6a0"]))
p = reg_cv["Gradient Boosting"].mean(0)
for s_ in sorted(grp.unique()):
    m = (grp == s_).values
    ax.scatter(y[m], p[m], s=22, color=cols[s_], label=s_, alpha=.85)
ax.plot([0, 100], [0, 100], "w--", lw=.8); ax.set_xlim(10, 100); ax.set_ylim(10, 100)
ax.set_xlabel("Measured yield (%)"); ax.set_ylabel("Predicted yield (%)")
ax.set_title("Random 5-fold CV (Gradient Boosting)", loc="left", fontweight="bold", fontsize=10)
ax.legend(ncol=2, fontsize=8, frameon=False, title="Study")
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
save(fig, "parity.png")

# tree importance (like previous project's Fig. 2)
top = imp_tree.head(15)[::-1]
fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=BG); ax.set_facecolor(BG)
cm = plt.cm.viridis(np.linspace(.15, .95, len(top)))
ax.barh(top.index, top.values, color=cm); ax.set_xlabel("Importance (Gradient Boosting, impurity-based)")
ax.set_title("Feature importance (one-hot encoded features)", loc="left", fontweight="bold")
for s in ("top", "right"): ax.spines[s].set_visible(False)
save(fig, "importance_tree.png")

# permutation importance
pp = perm.sort_values("MSE increase")
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG); ax.set_facecolor(BG)
yy = np.arange(len(pp))
ax.barh(yy, pp.iloc[:, 0], color="#8ab4f8")
ax.set_yticks(yy); ax.set_yticklabels(pp.index); ax.axvline(0, color="w", lw=.6)
ax.set_xlabel("Increase in test MSE when feature is shuffled")
ax.set_title("Permutation importance (original features)", loc="left", fontweight="bold")
for s in ("top", "right"): ax.spines[s].set_visible(False)
save(fig, "importance_perm.png")

# SHAP-style beeswarm
order = np.argsort(np.abs(sv).mean(0))            # ascending -> top feature last (plotted on top)
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=BG); ax.set_facecolor(BG)
for rank, j in enumerate(order):
    c = FEATURES[j]; v = X[c]
    if c in NUM:
        z = (v - v.min()) / (v.max() - v.min() + 1e-9); colr = plt.cm.coolwarm(z)
    else:
        colr = "#9e9e9e"
    ax.scatter(sv[:, j], rank + rng.uniform(-.3, .3, len(sv)), s=12, c=colr, alpha=.85, linewidths=0)
ax.set_yticks(range(len(order))); ax.set_yticklabels([NICE[FEATURES[j]] for j in order])
ax.axvline(0, color="w", lw=.6); ax.set_xlabel("Shapley value (impact on predicted yield, %-points)")
ax.set_title("Shapley-style feature impact (red = high value, blue = low, grey = categorical)", loc="left", fontweight="bold", fontsize=10)
sm = plt.cm.ScalarMappable(cmap="coolwarm"); sm.set_array([0, 1])
cb = fig.colorbar(sm, ax=ax, pad=.01); cb.set_ticks([0, 1]); cb.set_ticklabels(["Low", "High"]); cb.set_label("Feature value")
for s in ("top", "right"): ax.spines[s].set_visible(False)
save(fig, "shap_beeswarm.png")

ms = pd.Series(np.abs(sv).mean(0), index=[NICE[c] for c in FEATURES]).sort_values()
fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=BG); ax.set_facecolor(BG)
ax.barh(ms.index, ms.values, color=plt.cm.viridis(np.linspace(.15, .95, len(ms))))
ax.set_xlabel("Mean |Shapley value| (%-points of yield)"); ax.set_title("Global feature impact", loc="left", fontweight="bold")
for s in ("top", "right"): ax.spines[s].set_visible(False)
save(fig, "shap_bar.png")
ms.sort_values(ascending=False).to_csv(ART / "shap_mean_abs.csv", header=["mean_abs_shap"])

# PDP panel
fig, ax = plt.subplots(1, 5, figsize=(16, 3.2), facecolor=BG)
for a, c in zip(ax, NUM[1:]):
    a.set_facecolor(BG); g, v = pdp[c]; a.plot(g, v, color="#98e6a8", lw=2)
    a.set_xlabel(NICE[c]); a.set_ylim(55, 90)
    for s in ("top", "right"): a.spines[s].set_visible(False)
ax[0].set_ylabel("Avg predicted yield (%)")
fig.suptitle("Partial dependence of predicted yield (average over all literature conditions)", x=.01, ha="left", fontweight="bold")
fig.tight_layout(); save(fig, "pdp.png")

# ------------------------------------------------------------------ summary json
summ = {"n_rows": len(d), "n_studies": int(grp.nunique()), "best_regressor": best_reg,
        "high_yield_threshold": core.HIGH_YIELD_THRESHOLD, "high_yield_share": float(yc.mean()),
        "shap_base_value": base_val, "features": FEATURES}
json.dump(summ, open(ART / "summary.json", "w"), indent=2)
print("done", summ)
