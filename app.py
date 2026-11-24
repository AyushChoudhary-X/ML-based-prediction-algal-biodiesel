import io, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import core
from core import FEATURES, NUM, CAT, TARGET, NICE

st.set_page_config(page_title="Algal Biodiesel ML", page_icon="🌿", layout="wide")
ART = core.ROOT / "artifacts"; FIG = ART / "figs"
plt.style.use("dark_background")
BG = "#0e1117"


# ----------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading data and fitting final models…")
def load():
    d = core.load_data()
    return d, core.fit_final(d)


@st.cache_data
def csv(name):
    return pd.read_csv(ART / name)


d, models = load()
summary = json.load(open(ART / "summary.json"))
STUDY_DF = (d[d.Study_ID != "S11"].groupby("Study_ID")
            .agg(Species=("Algae_Species", "first"), Cultivation=("Cultivation_Mode", "first"),
                 Recovery=("Lipid_Recovery_Group", "first"), Catalyst=("Catalyst_Class", "first"),
                 Support=("Catalyst_/Support/", "first"), Lipid=("Biomass_Lipid_Content_wt%", "first"),
                 Runs=("Sl. no.", "count"), Ymin=(TARGET, "min"), Ymax=(TARGET, "max")).reset_index())
STUDY_DF["label"] = STUDY_DF.apply(lambda r: f"{r.Study_ID} · {r.Species} · {r.Catalyst} ({r.Support}) · {r.Runs} runs", axis=1)


def system_context(sid):
    sub = d[d.Study_ID == sid]
    r = sub.iloc[0]
    ctx = {c: r[c] for c in CAT}; ctx["Biomass_Lipid_Content_wt%"] = float(r["Biomass_Lipid_Content_wt%"])
    bounds = {c: (float(sub[c].min()), float(sub[c].max())) for c in NUM}
    return sub, ctx, bounds


def styled_best(df, cols, fmt="{:.3f}"):
    sty = df.style.format({c: fmt for c in cols}, na_rep="–")
    return sty.highlight_max(subset=cols, color="#2e7d32")


def pass_fail(df, cols):
    f = (lambda v: "background-color:#1b5e20" if v == "PASS" else "background-color:#7f1d1d")
    st_ = df.style
    return st_.map(f, subset=cols) if hasattr(st_, "map") else st_.applymap(f, subset=cols)


# ----------------------------------------------------------------------------- sidebar
PAGES = ["Overview", "Data Explorer", "Model Training", "Feature Importance & SHAP", "Predict Yield",
         "Process Optimiser", "Fuel Quality Check", "Catalyst Reusability", "Results & Conclusions"]
st.sidebar.title("🌿 Algal Biodiesel ML")
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption(f"{summary['n_rows']} experimental runs · {summary['n_studies']} studies · target: biodiesel yield (%)")

# ============================================================================= OVERVIEW
if page == "Overview":
    st.title("Machine-learning analysis of algal biodiesel production")
    st.write("Models trained on a curated literature dataset of **transesterification experiments on algal oil**. "
             "The target is **biodiesel yield (%)**; inputs are the algae/feedstock descriptors and the reaction conditions "
             "(alcohol:oil ratio, catalyst loading, temperature, time, catalyst reuse cycle).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experimental runs", len(d)); c2.metric("Literature studies", d.Study_ID.nunique())
    c3.metric("Mean yield", f"{d[TARGET].mean():.1f} %"); c4.metric("Max yield", f"{d[TARGET].max():.1f} %")
    st.subheader("Studies in the dataset")
    st.dataframe(STUDY_DF.drop(columns="label"), hide_index=True)
    st.info("Species, catalyst, lipid content and recovery method are fixed within each study, so the model mainly learns how "
            "reaction conditions change yield **inside the systems in the dataset**. Use predictions for those systems and stay "
            "within the tested ranges (the Predict page flags anything outside).")
    st.subheader("What you can do here")
    st.markdown("""
- **Model Training** – six models compared (regression *and* High/Low-yield classification).
- **Feature Importance & SHAP** – which reaction variables drive yield (permutation importance, Shapley-style attribution, partial dependence).
- **Predict Yield** – what-if prediction for a known system with an *extrapolation guard*, plus batch CSV prediction.
- **Process Optimiser** – search predicted-best conditions inside the range that was actually tested.
- **Fuel Quality Check** – each study's viscosity/density/cetane against ASTM D6751 and EN 14214.
- **Catalyst Reusability** – yield decay over reuse cycles.""")

# ============================================================================= DATA EXPLORER
elif page == "Data Explorer":
    st.title("Data explorer")
    f1, f2 = st.columns(2)
    sp = f1.multiselect("Algae species", sorted(d.Algae_Species.unique()), default=sorted(d.Algae_Species.unique()))
    cc = f2.multiselect("Catalyst class", sorted(d.Catalyst_Class.unique()), default=sorted(d.Catalyst_Class.unique()))
    v = d[d.Algae_Species.isin(sp) & d.Catalyst_Class.isin(cc)]
    st.caption(f"{len(v)} runs selected")
    t1, t2, t3, t4 = st.tabs(["Yield distribution", "Variable vs yield", "Correlations", "Table"])
    with t1:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
        ax[0].set_facecolor(BG); ax[0].hist(v[TARGET], bins=20, color="#8ab4f8"); ax[0].axvline(core.HIGH_YIELD_THRESHOLD, color="#ef5350", ls="--")
        ax[0].set_xlabel("Biodiesel yield (%)"); ax[0].set_title("Yield distribution (red = High-yield threshold)", loc="left", fontsize=10)
        ax[1].set_facecolor(BG)
        groups = [g[TARGET].values for _, g in v.groupby("Algae_Species")]
        if groups:
            ax[1].boxplot(groups, vert=False, tick_labels=[k for k, _ in v.groupby("Algae_Species")])
        ax[1].set_xlabel("Biodiesel yield (%)"); ax[1].set_title("Yield by species", loc="left", fontsize=10)
        fig.tight_layout(); st.pyplot(fig)
    with t2:
        xv = st.selectbox("X variable", NUM[1:] + ["Biomass_Lipid_Content_wt%"], format_func=lambda c: NICE[c])
        fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=BG); ax.set_facecolor(BG)
        for (sid, sp_), g in v.groupby(["Study_ID", "Algae_Species"]):
            ax.scatter(g[xv], g[TARGET], s=22, label=f"{sid} {sp_}", alpha=.85)
        ax.set_xlabel(NICE[xv]); ax.set_ylabel("Yield (%)"); ax.legend(fontsize=7, frameon=False)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        st.pyplot(fig)
    with t3:
        cols = NUM + [TARGET, "Kinematic_Viscosity_mm2_s", "Density_kg_m3", "Cetane_Number"]
        cm = v[cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
        im = ax.imshow(cm, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
        short = [NICE.get(c, c.replace("_", " ")) for c in cols]
        ax.set_xticklabels(short, rotation=60, ha="right", fontsize=7); ax.set_yticklabels(short, fontsize=7)
        for i in range(len(cols)):
            for j in range(len(cols)): ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax); fig.tight_layout(); st.pyplot(fig)
        st.caption("Fuel properties (viscosity, density, cetane) are nearly constant per study, so their correlations mostly reflect study differences.")
    with t4:
        st.dataframe(v.drop(columns=["High_Yield"]))
        st.download_button("Download filtered data (CSV)", v.to_csv(index=False).encode(), "filtered_data.csv")

# ============================================================================= MODEL TRAINING
elif page == "Model Training":
    st.title("Model training")
    st.write(f"Six algorithms were trained on **{len(d)} runs** and scored with **random 5-fold cross-validation, repeated 3 times**.")
    task = st.radio("Task", ["Classification: High vs Low yield", "Regression: predict yield (%)"], horizontal=True)
    if task.startswith("Class"):
        m = csv("metrics_classification.csv")
        st.caption(f"High-yield (HY) = yield ≥ {core.HIGH_YIELD_THRESHOLD:.0f} % ({summary['high_yield_share']*100:.0f} % of runs). Highlighted = best in column.")
        st.subheader("Table 1: Classification performance")
        st.dataframe(styled_best(m, [c for c in m.columns if c != "Model"]), hide_index=True)
        st.image(str(FIG / "cls_random.png"), caption="Figure 1: Accuracy and F1-score of the classifiers.")
    else:
        m = csv("metrics_regression.csv")
        st.subheader("Table 2: Regression performance")
        sty = m.style.format({"R2": "{:.3f}", "RMSE": "{:.2f}", "MAE": "{:.2f}"}, na_rep="–")
        sty = sty.highlight_max(subset=["R2"], color="#2e7d32").highlight_min(subset=["RMSE", "MAE"], color="#2e7d32")
        st.dataframe(sty, hide_index=True)
        c1, c2 = st.columns(2)
        c1.image(str(FIG / "reg_r2.png"), caption="Figure 1: R² of each regressor.")
        c2.image(str(FIG / "parity.png"), caption="Figure 2: Measured vs predicted yield (Gradient Boosting).")
    with st.expander("Modelling details"):
        st.markdown(f"""
- **Inputs (10):** {', '.join(NICE[c] for c in FEATURES)}. Categorical inputs are one-hot encoded; distance/linear models also standardise numeric inputs.
- **Final predictor:** average of Gradient Boosting (300 trees, depth 3, lr 0.05, subsample 0.8) and Random Forest (500 trees); the forest's tree-to-tree spread is shown as an uncertainty indicator.
- **Not used as inputs:** viscosity, density, cetane (they are outputs and almost constant per study); cloud point and total glycerol (dropped during cleaning, too sparse).
- **Cleaning notes that matter:** catalyst reuse cycle was set to 0 where no reuse was reported; cetane for one Ulva study is a donor fill; neither affects the yield target, which is unmodified.""")

# ============================================================================= IMPORTANCE & SHAP
elif page == "Feature Importance & SHAP":
    st.title("Feature importance and SHAP analysis")
    t1, t2, t3, t4 = st.tabs(["Tree importance", "Permutation importance", "SHAP-style analysis", "Partial dependence"])
    with t1:
        st.image(str(FIG / "importance_tree.png"), caption="Figure 1: Gradient Boosting impurity importance on one-hot features.")
    with t2:
        st.image(str(FIG / "importance_perm.png"), caption="Figure 2: Increase in test error when a feature is shuffled (higher = more important).")
        st.dataframe(csv("importance_permutation.csv").rename(columns={"Unnamed: 0": "Feature"}), hide_index=True)
    with t3:
        st.image(str(FIG / "shap_beeswarm.png"), caption="Figure 3: Feature impact distribution (Monte-Carlo Shapley values).")
        st.image(str(FIG / "shap_bar.png"), caption="Figure 4: Mean absolute Shapley value.")
        st.markdown("""
**How to read the beeswarm:** x-axis = impact on the predicted yield (right → pushes yield up, left → pushes it down). Red = high feature value, blue = low, grey = categorical.
*Example:* a cluster of blue dots far left on *Catalyst loading* means low loading is lowering predicted yield.

*Note:* these are Monte-Carlo permutation-sampling Shapley values computed on the blended final model (80 samples per run); they are approximate but model-agnostic, and the explanation describes the **model**, not proven chemistry.""")
    with t4:
        st.image(str(FIG / "pdp.png"), caption="Figure 5: Average predicted yield as each variable is swept across its observed range.")

# ============================================================================= PREDICT
elif page == "Predict Yield":
    st.title("Predict biodiesel yield")
    st.caption("Pick a literature system, then change the reaction conditions. Species/catalyst/lipid content are fixed by the system.")
    lab = st.selectbox("Literature system", STUDY_DF.label)
    sid = STUDY_DF.loc[STUDY_DF.label == lab, "Study_ID"].iloc[0]
    sub, ctx, bounds = system_context(sid)
    st.write(f"**Species:** {ctx['Algae_Species']} · **Catalyst:** {ctx['Catalyst_Class']} ({sub['Catalyst_/Support/'].iloc[0]}) · "
             f"**Lipid recovery:** {ctx['Lipid_Recovery_Group']} · **Lipid content:** {ctx['Biomass_Lipid_Content_wt%']} wt%")
    PROC = NUM[1:]
    cols = st.columns(len(PROC)); vals = {}
    for c, w in zip(PROC, cols):
        lo, hi = bounds[c]; med = float(sub[c].median())
        step = 1.0 if c in ("Reaction_Temperature_C", "Reaction_Time_min", "Catalyst_Reusability_Cycle") else 0.25
        vals[c] = w.number_input(NICE[c], min_value=0.0, value=med, step=step)
        w.caption(f"tested: {lo:g} – {hi:g}")
    row = pd.DataFrame([{**ctx, **vals}])
    r = core.predict(models, row)
    y0, sp0, ph = float(r["yield"][0]), float(r["spread"][0]), float(r["p_high"][0])
    out = [c for c in PROC if not (bounds[c][0] - 1e-9 <= vals[c] <= bounds[c][1] + 1e-9)]
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted yield", f"{y0:.1f} %"); m2.metric("Model spread (±)", f"{sp0:.1f}")
    m3.metric(f"P(yield ≥ {core.HIGH_YIELD_THRESHOLD:.0f}%)", f"{ph*100:.0f} %")
    if out:
        st.error("⚠️ Extrapolation: " + ", ".join(NICE[c] for c in out) + " outside the range tested for this system. "
                 "Tree models cannot extrapolate, so treat this number as unreliable.")
    elif sp0 > 12:
        st.warning("Large spread between trees: this region is sparsely sampled; interpret with care.")
    else:
        st.success("Inside the tested range for this system.")
    st.subheader("Closest measured runs in this system")
    Z = (sub[PROC] - sub[PROC].mean()) / (sub[PROC].std().replace(0, 1))
    zq = (pd.Series(vals) - sub[PROC].mean()) / (sub[PROC].std().replace(0, 1))
    near = sub.assign(Distance=np.sqrt(((Z - zq) ** 2).sum(1))).nsmallest(5, "Distance")
    st.dataframe(near[PROC + [TARGET, "Distance"]].rename(columns=NICE).round(2), hide_index=True)
    st.divider()
    st.subheader("Batch prediction (CSV)")
    tmpl = sub[FEATURES].head(3)
    st.download_button("Download CSV template", tmpl.to_csv(index=False).encode(), "template.csv")
    up = st.file_uploader("Upload a CSV with the template columns", type="csv")
    if up is not None:
        try:
            u = pd.read_csv(up); miss = [c for c in FEATURES if c not in u.columns]
            if miss: st.error(f"Missing columns: {miss}")
            else:
                rr = core.predict(models, u)
                u["Predicted_Yield_%"] = rr["yield"].round(1); u["Spread_±"] = rr["spread"].round(1)
                u["P_high_yield"] = rr["p_high"].round(2)
                st.dataframe(u); st.download_button("Download predictions", u.to_csv(index=False).encode(), "predictions.csv")
        except Exception as e:
            st.error(f"Could not process file: {e}")

# ============================================================================= OPTIMISER
elif page == "Process Optimiser":
    st.title("Process optimiser")
    st.caption("Random search (4,000 candidates) over reaction conditions **inside the range tested for the chosen system**.")
    lab = st.selectbox("Literature system", STUDY_DF.label)
    sid = STUDY_DF.loc[STUDY_DF.label == lab, "Study_ID"].iloc[0]
    sub, ctx, bounds = system_context(sid)
    PROC = NUM[1:]
    pins = st.multiselect("Fix these variables at their median (e.g. to limit energy use)", PROC, format_func=lambda c: NICE[c])
    c2 = dict(ctx); c2.update({p: float(sub[p].median()) for p in pins})
    if st.button("Search for best conditions"):
        b = {**bounds}
        top = core.optimise(models, c2, b, n=4000, top=8)
        st.dataframe(top[PROC + ["Predicted_Yield_%", "Uncertainty_±"]].rename(columns=NICE), hide_index=True)
        best = top.iloc[0]
        st.success(f"Best predicted yield: **{best['Predicted_Yield_%']:.1f} %** (measured max in this system: {sub[TARGET].max():.1f} %).")
        st.session_state["best"] = (sid, best[PROC].to_dict())
    if "best" in st.session_state and st.session_state["best"][0] == sid:
        st.subheader("What-if sweep around the best condition")
        base = st.session_state["best"][1]
        v = st.selectbox("Sweep variable", PROC, format_func=lambda c: NICE[c])
        lo, hi = bounds[v]
        if hi > lo:
            g = np.linspace(lo, hi, 40)
            B = pd.DataFrame([{**ctx, **base, v: x} for x in g])
            rr = core.predict(models, B)
            fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=BG); ax.set_facecolor(BG)
            ax.plot(g, rr["yield"], color="#98e6a8", lw=2)
            ax.fill_between(g, rr["yield"] - rr["spread"], rr["yield"] + rr["spread"], color="#98e6a8", alpha=.2)
            ax.set_xlabel(NICE[v]); ax.set_ylabel("Predicted yield (%)")
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            st.pyplot(fig)
        else:
            st.info("This variable was held constant in the source study.")
    st.caption("Optimum values are model suggestions inside observed ranges; confirm experimentally before use.")

# ============================================================================= FUEL QUALITY
elif page == "Fuel Quality Check":
    st.title("Fuel-quality screening")
    fc = core.fuel_check(d)
    st.dataframe(pass_fail(fc, ["ASTM D6751 (US)", "EN 14214 (EU)"]), hide_index=True)
    st.markdown("""
| Property | ASTM D6751 | EN 14214 |
|---|---|---|
| Kinematic viscosity @40 °C (mm²/s) | 1.9 – 6.0 | 3.5 – 5.0 |
| Cetane number | ≥ 47 | ≥ 51 |
| Density (kg/m³) | – | 860 – 900 @15 °C |""")
    st.caption("Values are study-level means reported in the source papers (measurement temperatures may differ), not model predictions. "
               "Cetane for Ulva study S4 is a donor value from study S5 (same species), flagged as lowest-confidence in the cleaning log. "
               "Properties could not be modelled from process conditions because they hardly vary within a study.")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), facecolor=BG)
    for a, (c, lim) in zip(ax, [("Kinematic_Viscosity_mm2_s", (3.5, 5.0)), ("Density_kg_m3", (860, 900)), ("Cetane_Number", (51, None))]):
        a.set_facecolor(BG); a.bar(fc.Study, fc[c], color="#8ab4f8")
        for l in lim:
            if l is not None: a.axhline(l, color="#ef5350", ls="--", lw=1)
        a.set_title(NICE.get(c, c.replace("_", " ")) + " (red = EN 14214)", loc="left", fontsize=9)
        if c == "Density_kg_m3": a.set_ylim(840, 900)
    fig.tight_layout(); st.pyplot(fig)

# ============================================================================= REUSABILITY
elif page == "Catalyst Reusability":
    st.title("Catalyst reusability")
    keys = ["Study_ID", "Catalyst_/Support/"] + [c for c in NUM if c != "Catalyst_Reusability_Cycle"]
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=BG); ax.set_facecolor(BG); n = 0
    rows = []
    for k, g in d.groupby(keys):
        g = g.sort_values("Catalyst_Reusability_Cycle")
        if g["Catalyst_Reusability_Cycle"].nunique() >= 3:
            lbl = f"{k[0]} {k[1][:22]}"
            ax.plot(g["Catalyst_Reusability_Cycle"], g[TARGET], marker="o", label=lbl); n += 1
            rows.append({"System": lbl, "Cycles": int(g["Catalyst_Reusability_Cycle"].max()),
                         "Yield first run (%)": g[TARGET].iloc[0], "Yield last run (%)": g[TARGET].iloc[-1],
                         "Loss (%-pts)": g[TARGET].iloc[0] - g[TARGET].iloc[-1]})
    ax.set_xlabel("Reuse cycle"); ax.set_ylabel("Yield (%)"); ax.legend(fontsize=7, frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    st.pyplot(fig)
    if rows: st.dataframe(pd.DataFrame(rows).round(1), hide_index=True)
    st.caption("Only series with ≥ 3 reuse cycles at otherwise identical conditions are shown. Studies disagree on whether the fresh run is cycle 0 or 1.")

# ============================================================================= RESULTS
else:
    st.title("Results & conclusions")
    reg = csv("metrics_regression.csv"); cls = csv("metrics_classification.csv")
    gb = reg[reg.Model == "Gradient Boosting"].iloc[0]; cg = cls[cls.Model == "Gradient Boosting"].iloc[0]
    sh = csv("shap_mean_abs.csv")
    top3 = ", ".join(sh.iloc[:3, 0].tolist())
    fc = core.fuel_check(d)
    n_astm = int((fc["ASTM D6751 (US)"] == "PASS").sum()); n_en_fail = int((fc["EN 14214 (EU)"] == "FAIL").sum())
    st.markdown(f"""
- **Algorithm performance:** Gradient Boosting was the strongest model (R² = {gb['R2']:.2f}, RMSE = {gb['RMSE']:.1f} %-pts; High/Low-yield accuracy = {cg['Accuracy']:.2f}, F1 = {cg['F1']:.2f}), followed by Random Forest. Tree ensembles edged out the kernel, distance-based, neural and linear models.
- **Key drivers (Shapley-style ranking):** {top3} carry the largest impact on predicted yield; categorical system descriptors matter less once the reaction variables are known.
- **Practical use:** the predictor and optimiser are intended for choosing new conditions within a studied catalyst/species system, with the extrapolation guard flagging anything outside tested ranges.
- **Fuel quality:** {n_astm} of {len(fc)} studies meet ASTM D6751 (viscosity, cetane); {n_en_fail} of {len(fc)} fall outside at least one EN 14214 limit (see Fuel Quality Check).
- **Next steps:** add more studies per catalyst class, include operating variables such as agitation and water content, and validate on newly published experiments.""")
