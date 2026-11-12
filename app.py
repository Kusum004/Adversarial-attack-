import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.decomposition import PCA

import aml_core as core

# ==========================================
# PAGE CONFIG & CYBERSECURITY THEMING
# ==========================================
st.set_page_config(
    page_title="AML Security & Defense Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #e2e8f0;
             font-family: 'Inter', system-ui, -apple-system, sans-serif; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.55), rgba(15, 23, 42, 0.75));
        border-radius: 12px;
    }
    .metric-card { background: rgba(15, 23, 42, 0.6); border-left: 4px solid #38bdf8;
                   border-radius: 8px; padding: 12px 15px; margin: 5px 0; }
    .metric-card h4 { margin: 0; font-size: 0.9rem; color: #94a3b8; }
    .metric-card h2 { margin: 4px 0; }
    .metric-card p { margin: 0; font-size: 0.8rem; color: #94a3b8; }
    .badge-clean { background-color: #065f46; color: #34d399; padding: 4px 10px;
                   border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .badge-attack { background-color: #881337; color: #fb7185; padding: 4px 10px;
                    border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 48px; white-space: pre-wrap; background-color: #1e293b;
                                   border-radius: 8px 8px 0 0; color: #94a3b8; padding: 10px 14px; }
    .stTabs [aria-selected="true"] { background-color: #0284c7 !important; color: #ffffff !important;
                                     font-weight: bold; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)", font_color="white")
BLUE, RED, GREEN, YELLOW, VIOLET = "#38bdf8", "#fb7185", "#34d399", "#facc15", "#a78bfa"


def pct(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}%"


def metric_card(title, value, subtitle, color=BLUE):
    st.markdown(f"<div class='metric-card' style='border-left-color:{color}'><h4>{title}</h4>"
                f"<h2 style='color:{color};'>{value}</h2><p>{subtitle}</p></div>", unsafe_allow_html=True)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# CACHED RESOURCES
# ==========================================
@st.cache_resource(show_spinner="Building fraud dataset and training tabular models…")
def get_tabular(seed):
    data = core.make_fraud_dataset(seed)
    models = core.train_tabular_models(data.X_train, data.y_train, seed)
    kit = core.build_defense_kit(data, seed)
    pca = PCA(n_components=2).fit(data.X_train)
    return data, models, kit, pca


@st.cache_resource(show_spinner="Loading the Fashion-MNIST CNN (training it if no saved weights exist)…")
def get_vision():
    device = torch.device("cpu")
    model, loaded = core.load_or_train_cnn(device)
    try:
        test_set = core.load_fashion_mnist(train=False)
        x_all, y_all = core.dataset_to_tensors(test_set)
        with torch.no_grad():
            test_acc = (model(x_all).argmax(1) == y_all).float().mean().item()
        source = "Fashion-MNIST test set"
    except Exception:  # offline: keep the app usable, but say so loudly
        x_all, y_all = torch.zeros(1000, 1, 28, 28), torch.zeros(1000, dtype=torch.long)
        test_acc, source = float("nan"), "UNAVAILABLE (download failed)"
    return model, x_all, y_all, test_acc, loaded, source


@st.cache_data(show_spinner="Running all 10 tabular attacks…")
def cached_tabular_summary(seed, eps, defense):
    data, models, kit, _ = get_tabular(seed)
    return core.tabular_summary(data, models, kit, eps, defense, seed)


@st.cache_data(show_spinner=False)
def cached_tabular_eval(seed, attack, target, eps, defense):
    data, models, kit, _ = get_tabular(seed)
    ctx = core.AttackContext(models, data.X_train, data.lo, data.hi, np.random.default_rng(seed))
    X_adv, target = core.run_tabular_attack(attack, data.X_test, data.y_test, eps, ctx, target)
    model = models[target]
    kind = core.TABULAR_ATTACKS[attack].kind
    if kind == "evasion":
        metrics = core.evaluate_classifier_attack(model, data.X_test, X_adv, data.y_test, defense, kit)
    elif target == "K-Means":
        metrics = core.evaluate_kmeans_attack(model, data.X_test, X_adv)
    else:
        metrics = core.evaluate_dbscan_attack(model, data.X_test, X_adv)
    return X_adv, metrics


@st.cache_data(show_spinner="Computing tabular robustness curves…")
def cached_tabular_curves(seed, attack, defense, eps_grid):
    data, models, kit, _ = get_tabular(seed)
    rows = []
    for name in core.CLASSIFIERS:
        for eps in eps_grid:
            ctx = core.AttackContext(models, data.X_train, data.lo, data.hi, np.random.default_rng(seed))
            X_adv, _ = core.run_tabular_attack(attack, data.X_test, data.y_test, eps, ctx, name)
            m = core.evaluate_classifier_attack(models[name], data.X_test, X_adv, data.y_test, defense, kit)
            rows.append({"Model": name, "ε": eps, "Undefended": m["adv_recall"], "Defended": m["def_adv_recall"]})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Attacking the evaluation batch…")
def cached_vision_eval(attack, eps, defenses, bits, sigma, n_eval, seed):
    model, x_all, y_all, *_ = get_vision()
    return core.evaluate_vision_attack(model, x_all[:n_eval], y_all[:n_eval], attack, eps,
                                       list(defenses), bits, sigma, seed)


@st.cache_data(show_spinner="Running all 10 vision attacks on the evaluation batch…")
def cached_vision_summary(eps, defenses, bits, sigma, n_eval, seed):
    rows = []
    for name in core.VISION_ATTACKS:
        m = cached_vision_eval(name, eps, defenses, bits, sigma, n_eval, seed)
        rows.append({"Attack": name, "Clean": m["clean_acc"], "Attacked": m["adv_acc"],
                     "Defended": m["def_adv_acc"], "Attack success": m["success_rate"],
                     "Mean L∞": m["linf"], "Mean L2": m["l2"]})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Computing vision robustness curves…")
def cached_vision_curves(defenses, bits, sigma, n_eval, seed, eps_grid):
    rows = []
    for attack in ["11. FGSM (Fast Gradient Sign Method)", "12. PGD (Projected Gradient Descent)"]:
        for eps in eps_grid:
            m = cached_vision_eval(attack, eps, defenses, bits, sigma, n_eval, seed)
            rows.append({"Attack": attack.split(" (")[0], "ε": eps,
                         "Undefended": m["adv_acc"], "Defended": m["def_adv_acc"]})
    return pd.DataFrame(rows)


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🛡️ AML Security Lab")
st.sidebar.markdown("**Course Project Dashboard**")
st.sidebar.markdown("---")
seed = st.sidebar.slider("🎲 Random Seed", 1, 100, 42,
                         help="Regenerates the fraud dataset, retrains the tabular models and reseeds every attack.")
n_eval = st.sidebar.select_slider("🖼️ Vision evaluation batch size", options=[128, 256, 512, 1000], value=256,
                                  help="Number of Fashion-MNIST test images used for batch metrics.")
st.sidebar.markdown("---")
st.sidebar.subheader("📌 System Benchmarks")
st.sidebar.markdown("- **Tabular Models**: 6 algorithms\n- **Vision Architecture**: PyTorch CNN\n"
                    "- **Tabular Attacks**: 10 techniques\n- **Vision Attacks**: 10 techniques\n"
                    "- **Defenses**: 4 tabular + 3 vision")
st.sidebar.markdown("---")
st.sidebar.caption("Threat model: defenses are tested against attacks crafted on the undefended model "
                   "(oblivious adversary). Adaptive attacks would be stronger.")

data, tab_models, defense_kit, pca = get_tabular(seed)
vision_model, x_all, y_all, cnn_test_acc, cnn_loaded, vision_source = get_vision()
class_names = core.CLASS_NAMES

# ==========================================
# HEADER & LIVE KPI ROW
# ==========================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🛡️ Adversarial Machine Learning Security Dashboard")
    st.markdown("*Attack & defense evaluation across tabular e-commerce fraud data and vision catalog data — "
                "every number on this page is computed live.*")
with col_h2:
    st.markdown("<div style='text-align: right; padding-top: 15px;'><span class='badge-clean'>"
                "Models trained</span><br><br><span class='badge-attack'>20 Attacks Available</span></div>",
                unsafe_allow_html=True)

KPI_TAB_ATTACK, KPI_TAB_EPS, KPI_TAB_DEF = "4. ZOO Finite Differences (GBM)", 0.35, "Feature Squeezing & Clipping Bounds"
KPI_VIS_ATTACK, KPI_VIS_EPS, KPI_VIS_DEF = "12. PGD (Projected Gradient Descent)", 0.15, ("Bit-Depth Reduction", "Spatial Median Blur")
_, kpi_tab = cached_tabular_eval(seed, KPI_TAB_ATTACK, "Random Forest", KPI_TAB_EPS, KPI_TAB_DEF)
kpi_vis = cached_vision_eval(KPI_VIS_ATTACK, KPI_VIS_EPS, KPI_VIS_DEF, 3, 0.25, n_eval, seed)

k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card("Tabular Baseline (RF)", pct(kpi_tab["clean_acc"]),
                f"Accuracy · fraud recall {pct(kpi_tab['clean_recall'])}")
with k2:
    metric_card("RF Recall under ZOO ε=0.35", f"{pct(kpi_tab['adv_recall'])} → {pct(kpi_tab['def_adv_recall'])}",
                "Attacked → with feature squeezing", GREEN)
with k3:
    metric_card("Vision CNN Baseline", pct(cnn_test_acc), f"Top-1 accuracy · {vision_source}")
with k4:
    metric_card("CNN under PGD ε=0.15", f"{pct(kpi_vis['adv_acc'])} → {pct(kpi_vis['def_adv_acc'])}",
                "Attacked → bit-depth + median defense", "#60a5fa")

if not cnn_loaded:
    st.info("No saved CNN weights were found, so the CNN was trained on first load and saved to `models/`.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Dataset & EDA Explorer",
    "🛡️ 2. Tabular AML Lab (Attacks 1-10)",
    "👁️ 3. Vision CNN Lab (Attacks 11-20)",
    "📈 4. Global Benchmark & Simulator",
])

# ==========================================
# TAB 1: DATASET & EDA
# ==========================================
with tab1:
    st.subheader("📊 Dataset Overview & Exploratory Data Analysis")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        with st.container(border=True):
            st.markdown("### 💳 Dataset 1: E-Commerce Fraud Transactions")
            st.caption("Synthetic data: fraud risk grows with large amounts, new accounts, night-time purchases "
                       "and large quantities, plus label noise. 15% of rows are fraud.")
            st.dataframe(data.df.head(10), width="stretch")
            st.markdown("**Feature statistics (raw units):**")
            st.dataframe(data.df[core.FEATURE_NAMES].describe().T[["mean", "std", "min", "50%", "max"]].round(2),
                         width="stretch")
    with col_d2:
        with st.container(border=True):
            st.markdown("### 🛍️ Dataset 2: Product Catalog Images (Fashion-MNIST)")
            fig_vis, axes_vis = plt.subplots(2, 5, figsize=(10, 4.4), facecolor="#0b0f19")
            with torch.no_grad():
                preds10 = vision_model(x_all[:10]).argmax(1)
            for i, ax in enumerate(axes_vis.flat):
                ax.imshow(x_all[i].squeeze().numpy(), cmap="gray", vmin=-1, vmax=1)
                ok = preds10[i].item() == y_all[i].item()
                ax.set_title(f"{class_names[y_all[i]]}\npred: {class_names[preds10[i]]}",
                             color=GREEN if ok else RED, fontsize=9)
                ax.axis("off")
            plt.tight_layout()
            show_fig(fig_vis)
            st.markdown("**Product categories**: " + ", ".join(class_names))

    with st.container(border=True):
        st.markdown("### 📈 Feature Correlation & Class Separation")
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            fig_corr = px.imshow(data.df.corr().round(2), text_auto=".2f", color_continuous_scale="RdBu_r",
                                 zmin=-1, zmax=1, title="Correlation Matrix")
            fig_corr.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig_corr, width="stretch")
        with col_e2:
            counts = data.df["is_fraud"].map({0: "Legitimate (0)", 1: "Fraud (1)"}).value_counts().reset_index()
            counts.columns = ["Class", "Count"]
            fig_class = px.bar(counts, x="Class", y="Count", color="Class",
                               color_discrete_map={"Legitimate (0)": BLUE, "Fraud (1)": RED},
                               title="Class Imbalance (15% fraud)")
            fig_class.update_layout(**PLOT_LAYOUT, showlegend=False)
            st.plotly_chart(fig_class, width="stretch")
        with col_e3:
            feat = st.selectbox("Feature distribution by class:", core.FEATURE_NAMES)
            fig_box = px.box(data.df.assign(Class=data.df["is_fraud"].map({0: "Legitimate", 1: "Fraud"})),
                             x="Class", y=feat, color="Class",
                             color_discrete_map={"Legitimate": BLUE, "Fraud": RED}, title=f"{feat} by class")
            fig_box.update_layout(**PLOT_LAYOUT, showlegend=False)
            st.plotly_chart(fig_box, width="stretch")

# ==========================================
# TAB 2: TABULAR AML LAB
# ==========================================
with tab2:
    st.subheader("🛡️ Tabular Adversarial Attack & Defense Sandbox (Attacks 1-10)")
    st.caption("Evasion attacks perturb only fraudulent transactions (the attacker is the fraudster), within an "
               "L∞ budget ε in standardized units and the feasible feature range seen in training.")
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        with st.container(border=True):
            st.markdown("### ⚙️ Experiment Settings")
            attack_name = st.selectbox("Adversarial attack technique:", list(core.TABULAR_ATTACKS))
            atk = core.TABULAR_ATTACKS[attack_name]
            st.caption(atk.note)
            if atk.agnostic:
                target_name = st.selectbox("Target classifier (black-box attack):", core.CLASSIFIERS,
                                           index=core.CLASSIFIERS.index(atk.target))
            else:
                target_name = atk.target
                st.markdown(f"**Target model:** {target_name} *(this attack needs its internals)*")
            eps_tab = st.slider("Perturbation budget ε (L∞, std. units)", 0.0, 2.0, 0.35, 0.05)
            st.markdown("### 🛡️ Defense Mechanism")
            enable_defense = st.checkbox("Enable tabular defense", value=True)
            defense_type = st.selectbox("Defense method:", core.TABULAR_DEFENSES, disabled=not enable_defense)
            active_defense = defense_type if enable_defense else "None"

    X_adv, tm = cached_tabular_eval(seed, attack_name, target_name, eps_tab, active_defense)
    with col_t2:
        with st.container(border=True):
            st.markdown(f"### 🎯 2D PCA View: `{attack_name}` → {target_name}")
            Xc, Xa = pca.transform(data.X_test), pca.transform(X_adv)
            moved = np.abs(X_adv - data.X_test).max(axis=1) > 1e-9
            fig_pca = go.Figure()
            fig_pca.add_trace(go.Scatter(x=Xc[data.y_test == 0, 0], y=Xc[data.y_test == 0, 1], mode="markers",
                                         marker=dict(size=5, color=BLUE, opacity=0.35), name="Legitimate (clean)"))
            fig_pca.add_trace(go.Scatter(x=Xc[data.y_test == 1, 0], y=Xc[data.y_test == 1, 1], mode="markers",
                                         marker=dict(size=7, color=VIOLET, opacity=0.8), name="Fraud (clean)"))
            fig_pca.add_trace(go.Scatter(x=Xa[moved, 0], y=Xa[moved, 1], mode="markers",
                                         marker=dict(size=7, color=RED, opacity=0.7), name="Attacked"))
            flipped = tm.get("evaded_mask", tm.get("flipped_mask"))
            if flipped is not None and flipped.any():
                fig_pca.add_trace(go.Scatter(
                    x=Xa[flipped, 0], y=Xa[flipped, 1], mode="markers",
                    marker=dict(size=12, color=YELLOW, symbol="star", line=dict(color="#ef4444", width=1)),
                    name=f"{'Evaded detection' if atk.kind == 'evasion' else 'Assignment changed'} ({flipped.sum()})"))
            fig_pca.update_layout(**PLOT_LAYOUT, xaxis_title="PC 1", yaxis_title="PC 2", height=430,
                                  margin=dict(t=30), legend=dict(orientation="h", y=1.08, x=0))
            st.plotly_chart(fig_pca, width="stretch")

            m1, m2, m3, m4 = st.columns(4)
            if atk.kind == "evasion":
                with m1:
                    metric_card("Clean recall", pct(tm["clean_recall"]), f"Accuracy {pct(tm['clean_acc'])}")
                with m2:
                    metric_card("Attacked recall", pct(tm["adv_recall"]), f"Accuracy {pct(tm['adv_acc'])}", RED)
                with m3:
                    metric_card("Defended recall", pct(tm["def_adv_recall"]) if enable_defense else "—",
                                f"Clean w/ defense {pct(tm['def_clean_recall'])}" if enable_defense else "Defense off",
                                GREEN)
                with m4:
                    metric_card("Evasion success", pct(tm["evasion_rate"]), "Detected fraud now missed", YELLOW)
            elif target_name == "K-Means":
                with m1:
                    metric_card("Clean silhouette", f"{tm['clean_sil']:.3f}", "Cluster separation")
                with m2:
                    metric_card("Attacked silhouette", f"{tm['adv_sil']:.3f}", "After perturbation", RED)
                with m3:
                    metric_card("Assignments changed", pct(tm["flip_rate"]), "Points switching cluster", YELLOW)
                with m4:
                    metric_card("Defense", "n/a", "Defenses here target classifiers", GREEN)
            else:
                with m1:
                    metric_card("Clean noise share", pct(tm["clean_noise"]), f"{tm['clean_clusters']} clusters")
                with m2:
                    metric_card("Attacked noise share", pct(tm["adv_noise"]), f"{tm['adv_clusters']} clusters", RED)
                with m3:
                    metric_card("Adjusted Rand index", f"{tm['ari']:.3f}", "Clean vs attacked labels", YELLOW)
                with m4:
                    metric_card("Defense", "n/a", "Defenses here target classifiers", GREEN)

    with st.container(border=True):
        st.markdown(f"### 📊 All 10 Attacks at ε = {eps_tab:.2f} · defense: {active_defense}")
        summary = cached_tabular_summary(seed, eps_tab, active_defense)
        st.dataframe(summary.style.format({c: "{:.3f}" for c in ["Clean", "Attacked", "Defended", "Attack success"]},
                                          na_rep="—"), width="stretch", hide_index=True)
        st.caption("Each attack runs against its default target. Attack success = share of correctly detected fraud "
                   "that evades (classifiers), share of points changing cluster (K-Means), or 1 − ARI (DBSCAN).")

# ==========================================
# TAB 3: VISION CNN LAB
# ==========================================
with tab3:
    st.subheader("👁️ Vision CNN Adversarial Attack & Defense Sandbox (Attacks 11-20)")
    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        with st.container(border=True):
            st.markdown("### 🖼️ Product Sample")
            sample_idx = st.selectbox("Catalog item:", list(range(20)),
                                      format_func=lambda i: f"Sample #{i + 1}: {class_names[y_all[i]]}")
            st.markdown("### ⚡ Vision Attack")
            vision_attack = st.selectbox("Attack (11-20):", list(core.VISION_ATTACKS))
            st.caption((core.VISION_ATTACKS[vision_attack].__doc__ or "").strip().split("\n")[0])
            v_eps = st.slider("Attack budget ε (pixel range is [-1, 1])", 0.0, 0.40, 0.15, 0.01)
            st.markdown("### 🛡️ Vision Defense Pipeline")
            v_defs = tuple(st.multiselect("Defenses (applied in order):", core.VISION_DEFENSES,
                                          default=["Bit-Depth Reduction", "Spatial Median Blur"]))
            bit_depth = st.slider("Bit-depth bits", 1, 8, 3, disabled="Bit-Depth Reduction" not in v_defs)
            sigma = st.slider("Smoothing noise σ", 0.05, 0.75, 0.25, 0.05,
                              disabled="Randomized Smoothing" not in v_defs)

    with col_v2:
        with st.container(border=True):
            st.markdown("### 🔬 Clean ➔ Attacked ➔ Defended")
            img = x_all[sample_idx:sample_idx + 1]
            lbl = y_all[sample_idx:sample_idx + 1]
            torch.manual_seed(seed)
            img_adv = core.VISION_ATTACKS[vision_attack](vision_model, img, lbl, v_eps) if v_eps > 0 else img.clone()
            img_seen = core.preprocess_for_display(img_adv, v_defs, bit_depth)
            with torch.no_grad():
                p_clean = torch.softmax(vision_model(img), 1)[0]
                p_adv = torch.softmax(vision_model(img_adv), 1)[0]
            p_def = core.defended_probs(vision_model, img_adv, v_defs, bit_depth, sigma)[0]
            true = lbl.item()

            fig3, ax3 = plt.subplots(1, 4, figsize=(13, 3.8), facecolor="#0b0f19")
            panels = [
                (img, f"1. Clean\ntrue: {class_names[true]}", p_clean, BLUE),
                (img_adv, "2. Attacked", p_adv, None),
                (img_seen, "3. Defended input", p_def, None),
            ]
            for ax, (im, title, p, col) in zip(ax3, panels):
                pred = int(p.argmax())
                col = col or (GREEN if pred == true else RED)
                ax.imshow(im[0].squeeze().numpy(), cmap="gray", vmin=-1, vmax=1)
                ax.set_title(f"{title}\npred: {class_names[pred]} ({p[pred] * 100:.1f}%)", color=col, fontsize=10)
                ax.axis("off")
            delta = (img_adv - img)[0].squeeze().numpy()
            lim = max(np.abs(delta).max(), 1e-6)
            ax3[3].imshow(delta, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax3[3].set_title(f"4. Perturbation\nL∞={np.abs(delta).max():.3f}  L2={np.linalg.norm(delta):.2f}",
                             color="white", fontsize=10)
            ax3[3].axis("off")
            plt.tight_layout()
            show_fig(fig3)

            prob_df = pd.DataFrame({"Class": class_names * 3,
                                    "Probability": np.concatenate([p_clean.numpy(), p_adv.numpy(), p_def.numpy()]),
                                    "Stage": ["Clean"] * 10 + ["Attacked"] * 10 + ["Defended"] * 10})
            fig_p = px.bar(prob_df, x="Class", y="Probability", color="Stage", barmode="group", height=280,
                           color_discrete_map={"Clean": BLUE, "Attacked": RED, "Defended": GREEN})
            fig_p.update_layout(**PLOT_LAYOUT, margin=dict(t=10), legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig_p, width="stretch")

    vm = cached_vision_eval(vision_attack, v_eps, v_defs, bit_depth, sigma, n_eval, seed)
    with st.container(border=True):
        st.markdown(f"### 📦 Batch Metrics on {n_eval} Test Images · `{vision_attack}`")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            metric_card("Clean accuracy", pct(vm["clean_acc"]),
                        f"With defense {pct(vm['def_clean_acc'])}" if v_defs else "No defense")
        with b2:
            metric_card("Attacked accuracy", pct(vm["adv_acc"]), f"Mean L∞ {vm['linf']:.3f}", RED)
        with b3:
            metric_card("Defended accuracy", pct(vm["def_adv_acc"]) if v_defs else "—",
                        ", ".join(v_defs) if v_defs else "Select a defense", GREEN)
        with b4:
            metric_card("Attack success", pct(vm["success_rate"]), "Correct predictions flipped", YELLOW)

    with st.container(border=True):
        st.markdown(f"### 🖼️ All 10 Attacks on '{class_names[true]}' (ε = {v_eps:.2f})")
        fig_g, axes_g = plt.subplots(2, 5, figsize=(14, 6), facecolor="#0b0f19")
        for ax, (name, fn) in zip(axes_g.flat, core.VISION_ATTACKS.items()):
            torch.manual_seed(seed)
            adv = fn(vision_model, img, lbl, v_eps) if v_eps > 0 else img
            with torch.no_grad():
                pi = vision_model(adv).argmax(1).item()
            ax.imshow(adv[0].squeeze().numpy(), cmap="gray", vmin=-1, vmax=1)
            ax.set_title(f"{name.split(' (')[0]}\n→ {class_names[pi]}", color=RED if pi != true else GREEN, fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        show_fig(fig_g)

    with st.container(border=True):
        st.markdown(f"### 📊 All 10 Vision Attacks on {n_eval} Images (ε = {v_eps:.2f})")
        vsum = cached_vision_summary(v_eps, v_defs, bit_depth, sigma, n_eval, seed)
        st.dataframe(vsum.style.format({"Clean": "{:.1%}", "Attacked": "{:.1%}", "Defended": "{:.1%}",
                                        "Attack success": "{:.1%}", "Mean L∞": "{:.3f}", "Mean L2": "{:.2f}"}),
                     width="stretch", hide_index=True)
        st.caption("JSMA (L0, ≈ ε·200 pixels saturated), the patch (unbounded inside its square), spatial "
                   "transforms and blur are not L∞-bounded, so their L∞ exceeds ε by design. Blur is a "
                   "non-adversarial corruption baseline.")

# ==========================================
# TAB 4: GLOBAL BENCHMARK & SIMULATOR
# ==========================================
with tab4:
    st.subheader("📈 Global Attack & Defense Benchmark")
    agnostic = [n for n, a in core.TABULAR_ATTACKS.items() if a.agnostic]
    g1, g2, g3 = st.columns(3)
    with g1:
        bench_attack = st.selectbox("Black-box tabular attack (applied to every classifier):", agnostic,
                                    index=agnostic.index("4. ZOO Finite Differences (GBM)"))
    with g2:
        bench_eps = st.slider("Tabular ε", 0.05, 2.0, 0.35, 0.05, key="bench_eps")
    with g3:
        bench_def = st.selectbox("Tabular defense:", core.TABULAR_DEFENSES, key="bench_def")

    with st.container(border=True):
        st.markdown("### 📊 Clean vs Attacked vs Defended (fraud recall for tabular, accuracy for CNN)")
        names, clean_v, adv_v, def_v = [], [], [], []
        for name in core.CLASSIFIERS:
            _, m = cached_tabular_eval(seed, bench_attack, name, bench_eps, bench_def)
            names.append(name)
            clean_v.append(m["clean_recall"])
            adv_v.append(m["adv_recall"])
            def_v.append(m["def_adv_recall"])
        names.append(f"CNN ({vision_attack.split(' (')[0]}, ε={v_eps:.2f})")
        clean_v.append(vm["clean_acc"])
        adv_v.append(vm["adv_acc"])
        def_v.append(vm["def_adv_acc"])
        fig_b = go.Figure([
            go.Bar(x=names, y=np.array(clean_v) * 100, name="Clean", marker_color=BLUE),
            go.Bar(x=names, y=np.array(adv_v) * 100, name="Attacked", marker_color=RED),
            go.Bar(x=names, y=np.array(def_v) * 100, name="Defended", marker_color=GREEN),
        ])
        fig_b.update_layout(**PLOT_LAYOUT, barmode="group", yaxis_title="%", yaxis_range=[0, 100])
        st.plotly_chart(fig_b, width="stretch")
        st.caption(f"Tabular bars use {bench_attack} at ε = {bench_eps:.2f} with {bench_def}. "
                   "The CNN bar uses the attack and defenses selected in Tab 3.")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("### 📉 Tabular Robustness Curve (fraud recall vs ε)")
            curves = cached_tabular_curves(seed, bench_attack, bench_def, (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5))
            long = curves.melt(id_vars=["Model", "ε"], var_name="Setting", value_name="Recall")
            fig_c = px.line(long, x="ε", y="Recall", color="Model", line_dash="Setting", markers=True)
            fig_c.update_layout(**PLOT_LAYOUT, yaxis_range=[0, 1], legend=dict(font=dict(size=10)))
            st.plotly_chart(fig_c, width="stretch")
    with c2:
        with st.container(border=True):
            st.markdown("### 📉 Vision Robustness Curve (accuracy vs ε)")
            vcurves = cached_vision_curves(v_defs, bit_depth, sigma, n_eval, seed,
                                           (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3))
            vlong = vcurves.melt(id_vars=["Attack", "ε"], var_name="Setting", value_name="Accuracy")
            fig_vc = px.line(vlong, x="ε", y="Accuracy", color="Attack", line_dash="Setting", markers=True,
                             color_discrete_sequence=[RED, YELLOW])
            fig_vc.update_layout(**PLOT_LAYOUT, yaxis_range=[0, 1])
            st.plotly_chart(fig_vc, width="stretch")
            st.caption(f"Defended = {', '.join(v_defs) if v_defs else 'no defense selected in Tab 3'}.")

    with st.container(border=True):
        st.markdown("### ⚡ Fraud Stress Simulator")
        st.write("Enter a transaction, then let a fraudster try to slip it past each model.")
        s1, s2, s3 = st.columns(3)
        with s1:
            sim_amt = st.number_input("Transaction amount ($)", 1.0, 5000.0, 450.0)
            sim_age = st.number_input("Customer age", 18, 100, 24)
        with s2:
            sim_acc = st.number_input("Account age (days)", 1, 3000, 12)
            sim_qty = st.number_input("Item quantity", 1, 20, 7)
        with s3:
            sim_hour = st.slider("Transaction hour", 0, 23, 2)
            sim_attack = st.checkbox(f"Fraudster applies {bench_attack} (ε = {bench_eps:.2f})", value=True)

        x_raw = np.array([[sim_amt, sim_age, sim_acc, sim_qty, sim_hour]], dtype=float)
        x_in = data.scaler.transform(x_raw)
        rows = []
        for name in core.CLASSIFIERS:
            model = tab_models[name]
            x_use = x_in
            if sim_attack:
                ctx = core.AttackContext(tab_models, data.X_train, data.lo, data.hi, np.random.default_rng(seed))
                x_use, _ = core.run_tabular_attack(bench_attack, x_in, np.array([1]), bench_eps, ctx, name)
            p_before = core.fraud_score(model, x_in)[0]
            p_after = core.fraud_score(model, x_use)[0]
            flagged = core.defended_predict(bench_def, model, x_use, defense_kit)[0]
            rows.append({"Model": name, "Fraud score (original)": p_before,
                         "Fraud score (after attack)": p_after if sim_attack else np.nan,
                         "Verdict (undefended)": "🚨 FRAUD" if model.predict(x_use)[0] == 1 else "✅ legit",
                         f"Verdict ({bench_def})": "🚨 FRAUD" if flagged == 1 else "✅ legit"})
        st.dataframe(pd.DataFrame(rows).style.format({"Fraud score (original)": "{:.1%}",
                                                      "Fraud score (after attack)": "{:.1%}"}, na_rep="—"),
                     width="stretch", hide_index=True)
        if sim_attack:
            adv_raw = data.scaler.inverse_transform(x_use)[0]
            st.caption("Last model's attacked transaction in raw units: " + ", ".join(
                f"{n} = {v:.1f}" for n, v in zip(core.FEATURE_NAMES, adv_raw)))

st.markdown("<br><hr><center><small>Built for AML CIA-3 Benchmark Evaluation</small></center>",
            unsafe_allow_html=True)
