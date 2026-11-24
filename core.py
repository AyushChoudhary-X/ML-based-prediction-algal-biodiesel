"""Shared logic: data prep, models, Shapley-style attribution, optimiser, fuel standards."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "algal_biodiesel_cleaned.xlsx"
TARGET = "Biodiesel_Yield_%"
HIGH_YIELD_THRESHOLD = 80.0          # >= 80 % -> "High Yield" (HY), else "Low Yield" (LY)
NUM = ["Biomass_Lipid_Content_wt%", "Alcohol_to_Oil_Molar_Ratio", "Catalyst_Loading_wt%",
       "Reaction_Temperature_C", "Reaction_Time_min", "Catalyst_Reusability_Cycle"]
CAT = ["Algae_Species", "Cultivation_Mode", "Lipid_Recovery_Group", "Catalyst_Class"]
FEATURES = NUM + CAT
NICE = {"Biomass_Lipid_Content_wt%": "Lipid content (wt%)", "Alcohol_to_Oil_Molar_Ratio": "Alcohol:oil molar ratio",
        "Catalyst_Loading_wt%": "Catalyst loading (wt%)", "Reaction_Temperature_C": "Reaction temp (°C)",
        "Reaction_Time_min": "Reaction time (min)", "Catalyst_Reusability_Cycle": "Catalyst reuse cycle",
        "Algae_Species": "Algae species", "Cultivation_Mode": "Cultivation mode",
        "Lipid_Recovery_Group": "Lipid recovery", "Catalyst_Class": "Catalyst class"}


# ----------------------------------------------------------------------------- data
def _cat_class(s):
    return "Homogeneous" if s.startswith("Homogeneous") else "Enzymatic" if s.startswith("Enzymatic") else "Heterogeneous"


def _recovery(s):
    s = s.lower()
    if "sc-co2" in s: return "SC-CO2 + ethanol"
    if "ultrasonic" in s: return "Ultrasound + microwave"
    if "methanol" in s: return "Soxhlet (methanol)"
    return "Soxhlet (hexane)"


def _cult(s):
    return ("Heterotrophic bioreactor" if s.startswith("Heterotrophic") else
            "Commercial biomass" if s.startswith("Commercial") else "Natural environment")


def load_data(path=DATA):
    d = pd.read_excel(path, sheet_name="Cleaned_Data")
    d["Catalyst_Class"] = d["Catalyst_Type"].map(_cat_class)
    d["Lipid_Recovery_Group"] = d["Lipid_Recovery_Method"].map(_recovery)
    d["Cultivation_Mode"] = d["Cultivation_Mode"].map(_cult)
    d["Algae_Species"] = d["Algae_Species"].replace({"Chlorophyceae (Chlorella) algae": "Chlorella (Chlorophyceae)"})
    d["High_Yield"] = (d[TARGET] >= HIGH_YIELD_THRESHOLD).astype(int)
    return d


# ----------------------------------------------------------------------------- models
def _pre(scale):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("num", num, NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])


REG = {"Random Forest": (lambda: RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1), False),
       "Gradient Boosting": (lambda: GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                                               subsample=0.8, random_state=0), False),
       "Ridge Regression": (lambda: Ridge(alpha=1.0), True),
       "SVR": (lambda: SVR(C=50, epsilon=1.0, gamma="scale"), True),
       "KNN": (lambda: KNeighborsRegressor(n_neighbors=5, weights="distance"), True),
       "ANN (MLP)": (lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=0), True)}
CLS = {"Random Forest": (lambda: RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1), False),
       "Gradient Boosting": (lambda: GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                                                subsample=0.8, random_state=0), False),
       "Logistic Regression": (lambda: LogisticRegression(max_iter=3000), True),
       "SVM": (lambda: SVC(C=10, probability=True, random_state=0), True),
       "KNN": (lambda: KNeighborsClassifier(n_neighbors=5, weights="distance"), True),
       "ANN (MLP)": (lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=0), True)}


def make_model(name, task="reg"):
    f, scale = (REG if task == "reg" else CLS)[name]
    return Pipeline([("pre", _pre(scale)), ("m", f())])


def fit_final(d):
    """Production models: Gradient Boosting + Random Forest (regression, averaged) and a GB classifier."""
    X, y = d[FEATURES], d[TARGET]
    return {"gb": make_model("Gradient Boosting").fit(X, y),
            "rf": make_model("Random Forest").fit(X, y),
            "clf": make_model("Gradient Boosting", "cls").fit(X, d["High_Yield"])}


def predict(models, X):
    X = X[FEATURES]
    gb = models["gb"].predict(X)
    rf = models["rf"].predict(X)
    Z = models["rf"][:-1].transform(X)
    trees = np.stack([t.predict(Z) for t in models["rf"]["m"].estimators_])
    return {"yield": np.clip(0.5 * gb + 0.5 * rf, 0, 100), "gb": gb, "rf": rf,
            "spread": trees.std(0), "p_high": models["clf"].predict_proba(X)[:, 1]}


class BlendModel:
    """GB + RF average, exposing .predict so it can be explained like any other model."""
    def __init__(self, models): self.m = models
    def predict(self, X): return np.clip(0.5 * self.m["gb"].predict(X[FEATURES]) + 0.5 * self.m["rf"].predict(X[FEATURES]), 0, 100)


# ----------------------------------------------------------------------------- attribution
def shapley_mc(model, X, n_perm=40, seed=0):
    """Monte-Carlo permutation-sampling Shapley values on the ORIGINAL features
    (model-agnostic, approximate; values sum to f(x) - E[f] in expectation)."""
    rng = np.random.default_rng(seed)
    F = list(X.columns); nF = len(F); n = len(X)
    Xv = X.reset_index(drop=True)
    vals = np.zeros((n, nF))
    base = float(np.mean(model.predict(Xv)))
    for i in range(n):
        rows, orders = [], []
        for _ in range(n_perm):
            z = Xv.iloc[rng.integers(n)].to_dict()
            order = rng.permutation(nF); orders.append(order)
            rows.append(dict(z))
            for j in order:
                z[F[j]] = Xv.at[i, F[j]]
                rows.append(dict(z))
        B = pd.DataFrame(rows)
        for c in NUM:
            if c in B: B[c] = B[c].astype(float)
        p = np.asarray(model.predict(B[F])).reshape(n_perm, nF + 1)
        for k, order in enumerate(orders):
            vals[i, order] += np.diff(p[k])
    return base, vals / n_perm


def partial_dependence(model, X, col, grid=30):
    g = np.linspace(X[col].min(), X[col].max(), grid)
    B = X.copy(); out = []
    for v in g:
        B[col] = v; out.append(np.mean(model.predict(B)))
    return g, np.array(out)


# ----------------------------------------------------------------------------- optimiser
def optimise(models, context: dict, bounds: dict, n=4000, top=8, seed=1):
    """Random search over numeric process conditions with the categorical context fixed."""
    rng = np.random.default_rng(seed)
    B = pd.DataFrame({c: rng.uniform(*bounds[c], n) for c in NUM})
    for c in ["Reaction_Temperature_C", "Reaction_Time_min", "Catalyst_Reusability_Cycle"]:
        B[c] = B[c].round()
    for c in CAT: B[c] = context[c]
    for c in NUM:
        if c in context: B[c] = context[c]           # pinned variable
    r = predict(models, B)
    B["Predicted_Yield_%"] = r["yield"].round(1); B["Uncertainty_±"] = r["spread"].round(1)
    return B.sort_values("Predicted_Yield_%", ascending=False).drop_duplicates(NUM).head(top).reset_index(drop=True)


# ----------------------------------------------------------------------------- fuel standards
STANDARDS = {
    "ASTM D6751 (US)": {"Kinematic_Viscosity_mm2_s": (1.9, 6.0), "Cetane_Number": (47, None), "Density_kg_m3": (None, None)},
    "EN 14214 (EU)": {"Kinematic_Viscosity_mm2_s": (3.5, 5.0), "Cetane_Number": (51, None), "Density_kg_m3": (860, 900)},
}


def _ok(v, lo, hi):
    return (lo is None or v >= lo) and (hi is None or v <= hi)


def fuel_check(d):
    rows = []
    for sid, g in d.groupby("Study_ID"):
        r = {"Study": sid, "Species": g["Algae_Species"].iloc[0], "Runs": len(g)}
        for col in ["Kinematic_Viscosity_mm2_s", "Density_kg_m3", "Cetane_Number"]:
            r[col] = round(g[col].mean(), 2)
        for std, lim in STANDARDS.items():
            r[std] = "PASS" if all(_ok(g[c].mean(), *l) for c, l in lim.items()) else "FAIL"
        rows.append(r)
    return pd.DataFrame(rows)
