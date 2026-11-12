"""
Core logic for the AML Security & Defense Lab.

Everything here is independent of Streamlit so it can be imported by the
dashboard (app.py), the training script (train_cnn.py) and the tests.

Threat model used throughout
----------------------------
* Tabular (fraud detection): the attacker is a fraudster who perturbs the
  features of *fraudulent* transactions to evade detection, within an
  L-infinity budget ``eps`` (in standardized feature units) and inside the
  feasible feature range seen in training data.
* Vision (catalog classification): untargeted misclassification of product
  images in normalized [-1, 1] pixel space, L-infinity budget ``eps`` unless
  an attack states otherwise (JSMA = L0, patch = physical patch).
* Defenses are evaluated against attacks crafted on the *undefended* model
  (an "oblivious" adversary). Adaptive attacks would be stronger.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, recall_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fashion_cnn.pt")

FEATURE_NAMES = ["transaction_amount", "customer_age", "account_age_days", "quantity", "transaction_hour"]
CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
CLASSIFIERS = ["Logistic Regression", "Support Vector Machine", "Random Forest", "Gradient Boosting"]


# =============================================================================
# TABULAR DATA & MODELS
# =============================================================================
@dataclass
class TabularData:
    df: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    lo: np.ndarray  # feasible lower bound per feature (standardized, from train)
    hi: np.ndarray  # feasible upper bound per feature


def make_fraud_dataset(seed: int = 42, n_samples: int = 2000, fraud_rate: float = 0.15) -> TabularData:
    """Synthetic e-commerce transactions whose fraud label depends on the features.

    Fraud risk rises with large amounts, new accounts, night-time purchases,
    large quantities and (slightly) younger customers, plus label noise, so
    models have a real but imperfect signal to learn. Exactly ``fraud_rate``
    of rows are labelled fraud.
    """
    rng = np.random.default_rng(seed)
    amount = rng.exponential(scale=100, size=n_samples)
    age = rng.integers(18, 70, size=n_samples)
    account_age = rng.integers(1, 1000, size=n_samples)
    quantity = rng.integers(1, 10, size=n_samples)
    hour = rng.integers(0, 24, size=n_samples)

    def z(v):
        v = np.asarray(v, dtype=float)
        return (v - v.mean()) / v.std()

    night = ((hour <= 5) | (hour >= 23)).astype(float)
    risk = (1.3 * z(np.log1p(amount)) - 1.1 * z(np.log1p(account_age)) + 1.2 * night
            + 0.5 * z(quantity) - 0.3 * z(age) + rng.normal(0, 0.7, size=n_samples))
    is_fraud = (risk > np.quantile(risk, 1 - fraud_rate)).astype(int)

    df = pd.DataFrame({
        "transaction_amount": amount.round(2),
        "customer_age": age,
        "account_age_days": account_age,
        "quantity": quantity,
        "transaction_hour": hour,
        "is_fraud": is_fraud,
    })
    X = df[FEATURE_NAMES].values.astype(float)
    y = df["is_fraud"].values
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_tr_raw)  # fit on train only: no test leakage
    X_train, X_test = scaler.transform(X_tr_raw), scaler.transform(X_te_raw)
    return TabularData(df, X_train, X_test, y_train, y_test, scaler,
                       lo=X_train.min(axis=0), hi=X_train.max(axis=0))


def train_tabular_models(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed).fit(X_train, y_train),
        "Support Vector Machine": SVC(kernel="rbf", random_state=seed).fit(X_train, y_train),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed).fit(X_train, y_train),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed).fit(X_train, y_train),
        "K-Means": KMeans(n_clusters=2, n_init=10, random_state=seed).fit(X_train),
        # DBSCAN has no predict(); it is re-fit on clean vs attacked data during evaluation.
        "DBSCAN": DBSCAN(eps=1.0, min_samples=5),
    }


def fraud_score(model, X: np.ndarray) -> np.ndarray:
    """Fraud probability (or a monotone sigmoid score for SVM)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return 1.0 / (1.0 + np.exp(-model.decision_function(X)))


# =============================================================================
# TABULAR ATTACKS (1-10)
# =============================================================================
@dataclass
class AttackContext:
    models: dict
    X_train: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))


def _project(X_adv, X, eps, lo, hi):
    """Project onto the L-inf eps-ball around X and the feasible feature box."""
    return np.clip(np.clip(X_adv, X - eps, X + eps), lo, hi)


def _atk_fgsm_linear(model, X, eps, ctx):
    # Exact FGSM for a linear model: the gradient sign of the fraud logit is sign(w).
    return X - eps * np.sign(model.coef_[0])


def _atk_greedy_threshold(model, X, eps, ctx):
    # Greedy coordinate search: per feature, try -eps / 0 / +eps and keep what lowers the score.
    X_adv = X.copy()
    order = np.argsort(-getattr(model, "feature_importances_", np.ones(X.shape[1])))
    for j in order:
        best = fraud_score(model, X_adv)
        for s in (-eps, eps):
            cand = X_adv.copy()
            cand[:, j] = X[:, j] + s
            cand = _project(cand, X, eps, ctx.lo, ctx.hi)
            sc = fraud_score(model, cand)
            better = sc < best
            X_adv[better], best[better] = cand[better], sc[better]
    return X_adv


def _atk_boundary_search(model, X, eps, ctx):
    # Decision-based (HopSkipJump-style): head toward the nearest training point the
    # model calls legitimate, then binary-search for the smallest step that flips it.
    legit = ctx.X_train[model.predict(ctx.X_train) == 0]
    idx = NearestNeighbors(n_neighbors=1).fit(legit).kneighbors(X, return_distance=False)[:, 0]
    d = legit[idx] - X
    linf = np.abs(d).max(axis=1, keepdims=True)
    d = d * np.minimum(1.0, eps / np.maximum(linf, 1e-12))
    crosses = model.predict(_project(X + d, X, eps, ctx.lo, ctx.hi)) == 0
    lo_t, hi_t = np.zeros(len(X)), np.ones(len(X))
    for _ in range(10):
        mid = (lo_t + hi_t) / 2
        ok = model.predict(_project(X + mid[:, None] * d, X, eps, ctx.lo, ctx.hi)) == 0
        hi_t, lo_t = np.where(ok, mid, hi_t), np.where(ok, lo_t, mid)
    t = np.where(crosses, hi_t, 1.0)
    return X + t[:, None] * d


def _atk_zoo(model, X, eps, ctx, iters=10, h=0.2):
    # Zeroth-order optimisation: finite-difference gradient estimate + signed descent.
    X_adv, step, n_feat = X.copy(), eps / 4, X.shape[1]
    for _ in range(iters):
        grad = np.zeros_like(X)
        for j in range(n_feat):
            e = np.zeros(n_feat)
            e[j] = h
            grad[:, j] = (fraud_score(model, X_adv + e) - fraud_score(model, X_adv - e)) / (2 * h)
        X_adv = _project(X_adv - step * np.sign(grad), X, eps, ctx.lo, ctx.hi)
    return X_adv


def _atk_newtonfool(model, X, eps, ctx, margin=0.5):
    # Newton step on the logit of a linear model: minimal L2 move to logit = -margin.
    w, b = model.coef_[0], model.intercept_[0]
    logit = X @ w + b
    step = np.where(logit > -margin, (-margin - logit) / (w @ w), 0.0)
    return X + step[:, None] * w


def _atk_random_walk(model, X, eps, ctx, iters=30):
    # Score-based random walk: accept random moves that lower the fraud score.
    X_adv, best = X.copy(), fraud_score(model, X)
    for _ in range(iters):
        cand = _project(X_adv + ctx.rng.normal(0, eps / 2, size=X.shape), X, eps, ctx.lo, ctx.hi)
        sc = fraud_score(model, cand)
        better = sc < best
        X_adv[better], best[better] = cand[better], sc[better]
    return X_adv


def _atk_uniform_noise(model, X, eps, ctx):
    return X + ctx.rng.uniform(-eps, eps, size=X.shape)


def _atk_centroid_displacement(model, X, eps, ctx):
    # Push each point toward the nearest *other* K-Means centroid.
    centers = model.cluster_centers_
    dist = ((X[:, None, :] - centers[None]) ** 2).sum(-1)
    own = dist.argmin(1)
    dist[np.arange(len(X)), own] = np.inf
    other = centers[dist.argmin(1)]
    return X + eps * np.sign(other - X)


def _atk_density_noise(model, X, eps, ctx):
    return X + ctx.rng.normal(0, eps, size=X.shape)


def _atk_shadow_transfer(model, X, eps, ctx):
    # Black-box transfer: query the target on attacker data, fit a surrogate, FGSM on it.
    surrogate = LogisticRegression(max_iter=1000).fit(ctx.X_train, model.predict(ctx.X_train))
    return X - eps * np.sign(surrogate.coef_[0])


@dataclass(frozen=True)
class TabularAttack:
    target: str        # default target model
    kind: str          # "evasion" (fraud rows only) or "clustering" (all rows)
    agnostic: bool     # can be pointed at any classifier (black-box)
    fn: object
    note: str


TABULAR_ATTACKS = {
    "1. Feature Manipulation / FGSM (LR)": TabularAttack(
        "Logistic Regression", "evasion", False, _atk_fgsm_linear,
        "White-box: moves every feature by ε against the sign of the LR weights."),
    "2. Threshold Bypassing (GBM)": TabularAttack(
        "Gradient Boosting", "evasion", True, _atk_greedy_threshold,
        "Score-based greedy coordinate search to slip under tree split thresholds."),
    "3. HopSkipJump Boundary Search (RF)": TabularAttack(
        "Random Forest", "evasion", True, _atk_boundary_search,
        "Decision-only: binary search toward the nearest legitimate-looking point."),
    "4. ZOO Finite Differences (GBM)": TabularAttack(
        "Gradient Boosting", "evasion", True, _atk_zoo,
        "Zeroth-order: estimates gradients from score queries, then signed descent."),
    "5. NewtonFool (LR)": TabularAttack(
        "Logistic Regression", "evasion", False, _atk_newtonfool,
        "White-box Newton step to the decision boundary, clipped to ε."),
    "6. Boundary Random Walk (SVM)": TabularAttack(
        "Support Vector Machine", "evasion", True, _atk_random_walk,
        "Score-based random walk accepting moves that lower the fraud score."),
    "7. Uniform Random Noise (K-Means)": TabularAttack(
        "K-Means", "clustering", False, _atk_uniform_noise,
        "Untargeted uniform noise in [-ε, ε] on every feature."),
    "8. Centroid Displacement (K-Means)": TabularAttack(
        "K-Means", "clustering", False, _atk_centroid_displacement,
        "Pushes each point toward the nearest other centroid."),
    "9. Core-Point Density Modification (DBSCAN)": TabularAttack(
        "DBSCAN", "clustering", False, _atk_density_noise,
        "Gaussian jitter that thins dense regions and breaks core points."),
    "10. Shadow-Model Transfer (RF)": TabularAttack(
        "Random Forest", "evasion", True, _atk_shadow_transfer,
        "Black-box transfer: surrogate LR trained on the target's answers."),
}


def run_tabular_attack(name, X, y, eps, ctx: AttackContext, target: str | None = None):
    """Return (X_adv, target_model_name). Evasion attacks only modify fraud rows."""
    atk = TABULAR_ATTACKS[name]
    target = target or atk.target
    model = ctx.models[target]
    X_adv = X.copy()
    if eps <= 0:
        return X_adv, target
    if atk.kind == "evasion":
        mask = y == 1
        if mask.any():
            X_adv[mask] = atk.fn(model, X[mask], eps, ctx)
    else:
        X_adv = atk.fn(model, X, eps, ctx)
    return _project(X_adv, X, eps, ctx.lo, ctx.hi), target


# =============================================================================
# TABULAR DEFENSES
# =============================================================================
TABULAR_DEFENSES = [
    "Feature Squeezing & Clipping Bounds",
    "Isolation Forest Outlier Purging",
    "KNN Neighborhood Consensus",
    "Quantile Sanitization",
]


@dataclass
class TabularDefenseKit:
    lo: np.ndarray
    hi: np.ndarray
    q_lo: np.ndarray
    q_hi: np.ndarray
    iso: IsolationForest
    knn: NearestNeighbors
    X_train: np.ndarray


def build_defense_kit(data: TabularData, seed: int = 42) -> TabularDefenseKit:
    return TabularDefenseKit(
        lo=data.lo, hi=data.hi,
        q_lo=np.percentile(data.X_train, 2.5, axis=0),
        q_hi=np.percentile(data.X_train, 97.5, axis=0),
        iso=IsolationForest(contamination=0.05, random_state=seed).fit(data.X_train),
        knn=NearestNeighbors(n_neighbors=7).fit(data.X_train),
        X_train=data.X_train,
    )


def defended_predict(defense: str | None, model, X: np.ndarray, kit: TabularDefenseKit) -> np.ndarray:
    """Predictions after the chosen input-sanitisation / detection defense."""
    if not defense or defense == "None":
        return model.predict(X)
    if defense == "Feature Squeezing & Clipping Bounds":
        return model.predict(np.round(np.clip(X, kit.lo, kit.hi) / 0.25) * 0.25)
    if defense == "Isolation Forest Outlier Purging":
        # Anomalous transactions are escalated (treated as fraud) instead of auto-approved.
        pred = model.predict(X)
        pred[kit.iso.predict(X) == -1] = 1
        return pred
    if defense == "KNN Neighborhood Consensus":
        idx = kit.knn.kneighbors(X, return_distance=False)
        return model.predict(kit.X_train[idx].mean(axis=1))
    if defense == "Quantile Sanitization":
        return model.predict(np.clip(X, kit.q_lo, kit.q_hi))
    raise ValueError(f"Unknown defense: {defense}")


# =============================================================================
# TABULAR EVALUATION
# =============================================================================
def evaluate_classifier_attack(model, X, X_adv, y, defense, kit) -> dict:
    p_clean, p_adv = model.predict(X), model.predict(X_adv)
    d_clean, d_adv = defended_predict(defense, model, X, kit), defended_predict(defense, model, X_adv, kit)
    detected = (y == 1) & (p_clean == 1)
    evaded = detected & (p_adv == 0)
    return {
        "clean_acc": accuracy_score(y, p_clean),
        "adv_acc": accuracy_score(y, p_adv),
        "def_clean_acc": accuracy_score(y, d_clean),
        "def_adv_acc": accuracy_score(y, d_adv),
        "clean_recall": recall_score(y, p_clean, zero_division=0),
        "adv_recall": recall_score(y, p_adv, zero_division=0),
        "def_clean_recall": recall_score(y, d_clean, zero_division=0),
        "def_adv_recall": recall_score(y, d_adv, zero_division=0),
        "evasion_rate": evaded.sum() / max(detected.sum(), 1),
        "evaded_mask": evaded,
    }


def evaluate_kmeans_attack(model, X, X_adv) -> dict:
    a_clean, a_adv = model.predict(X), model.predict(X_adv)
    return {
        "clean_sil": silhouette_score(X, a_clean),
        "adv_sil": silhouette_score(X_adv, a_adv) if len(set(a_adv)) > 1 else float("nan"),
        "flip_rate": float(np.mean(a_clean != a_adv)),
        "flipped_mask": a_clean != a_adv,
    }


def evaluate_dbscan_attack(model, X, X_adv) -> dict:
    l_clean = DBSCAN(**model.get_params()).fit_predict(X)
    l_adv = DBSCAN(**model.get_params()).fit_predict(X_adv)
    n_clusters = lambda l: len(set(l) - {-1})
    return {
        "clean_noise": float(np.mean(l_clean == -1)),
        "adv_noise": float(np.mean(l_adv == -1)),
        "clean_clusters": n_clusters(l_clean),
        "adv_clusters": n_clusters(l_adv),
        "ari": adjusted_rand_score(l_clean, l_adv),
        "flipped_mask": (l_clean != -1) & (l_adv == -1),
    }


def tabular_summary(data: TabularData, models: dict, kit, eps: float, defense: str, seed: int) -> pd.DataFrame:
    """Every attack run against its own default target at the same ε and defense."""
    rows = []
    for name, atk in TABULAR_ATTACKS.items():
        ctx = AttackContext(models, data.X_train, data.lo, data.hi, np.random.default_rng(seed))
        X_adv, target = run_tabular_attack(name, data.X_test, data.y_test, eps, ctx)
        model = models[target]
        if atk.kind == "evasion":
            m = evaluate_classifier_attack(model, data.X_test, X_adv, data.y_test, defense, kit)
            rows.append({"Attack": name, "Target": target, "Metric": "Fraud recall",
                         "Clean": m["clean_recall"], "Attacked": m["adv_recall"],
                         "Defended": m["def_adv_recall"], "Attack success": m["evasion_rate"]})
        elif target == "K-Means":
            m = evaluate_kmeans_attack(model, data.X_test, X_adv)
            rows.append({"Attack": name, "Target": target, "Metric": "Silhouette",
                         "Clean": m["clean_sil"], "Attacked": m["adv_sil"],
                         "Defended": np.nan, "Attack success": m["flip_rate"]})
        else:
            m = evaluate_dbscan_attack(model, data.X_test, X_adv)
            rows.append({"Attack": name, "Target": target, "Metric": "Non-noise share",
                         "Clean": 1 - m["clean_noise"], "Attacked": 1 - m["adv_noise"],
                         "Defended": np.nan, "Attack success": 1 - m["ari"]})
    return pd.DataFrame(rows)


# =============================================================================
# VISION MODEL
# =============================================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, num_classes),
        )

    def embed(self, x):
        """Penultimate-layer features (used by the feature-space attack)."""
        return self.classifier[:2](self.features(x).flatten(1))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def load_fashion_mnist(train: bool):
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return datasets.FashionMNIST(root=DATA_ROOT, train=train, download=True, transform=tf)


def dataset_to_tensors(ds, n: int | None = None):
    n = len(ds) if n is None else min(n, len(ds))
    x = (ds.data[:n].float() / 255.0 - 0.5) / 0.5
    return x.unsqueeze(1), ds.targets[:n].clone()


def train_cnn(epochs: int = 4, seed: int = 0, device="cpu", log=print) -> SimpleCNN:
    torch.manual_seed(seed)
    x, y = dataset_to_tensors(load_fashion_mnist(train=True))
    model = SimpleCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(x)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            xb, yb = x[idx].to(device), y[idx].to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        log(f"epoch {ep + 1}/{epochs}  train loss {total / n:.4f}")
    model.eval()
    return model


def load_or_train_cnn(device="cpu", log=print) -> tuple[SimpleCNN, bool]:
    """Return (model, loaded_from_disk). Trains and saves weights if none exist."""
    model = SimpleCNN().to(device)
    loaded = os.path.exists(MODEL_PATH)
    if loaded:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    else:
        model = train_cnn(device=device, log=log)
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
        except OSError:
            pass
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, loaded


# =============================================================================
# VISION ATTACKS (11-20)
# =============================================================================
def _loss_grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    g, = torch.autograd.grad(F.cross_entropy(model(x), y), x)
    return g


def _clip_ball(x_adv, x, eps):
    return torch.clamp(torch.max(torch.min(x_adv, x + eps), x - eps), -1.0, 1.0)


def attack_fgsm(model, x, y, eps):
    return torch.clamp(x + eps * _loss_grad(model, x, y).sign(), -1, 1).detach()


def attack_pgd(model, x, y, eps, iters=10):
    alpha = eps / 4
    x_adv = _clip_ball(x + torch.empty_like(x).uniform_(-eps, eps), x, eps)
    for _ in range(iters):
        x_adv = _clip_ball(x_adv + alpha * _loss_grad(model, x_adv, y).sign(), x, eps)
    return x_adv.detach()


def attack_cw(model, x, y, eps, steps=40, c=5.0, lr=0.05):
    """Carlini-Wagner L2 objective (margin loss + L2 penalty), kept inside the ε-ball."""
    w = torch.atanh(torch.clamp(x, -1 + 1e-6, 1 - 1e-6)).detach().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    onehot = F.one_hot(y, 10).bool()
    for _ in range(steps):
        x_adv = _clip_ball(torch.tanh(w), x, eps)
        logits = model(x_adv)
        real = logits[onehot]
        other = logits.masked_fill(onehot, -1e9).max(1).values
        loss = (((x_adv - x) ** 2).flatten(1).sum(1) + c * torch.clamp(real - other, min=0)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return _clip_ball(torch.tanh(w), x, eps).detach()


def attack_jsma(model, x, y, eps, rounds=5):
    """Saliency-map L0 attack: saturates the ~ε·200 most salient pixels."""
    budget = max(1, int(round(eps * 200)))
    per_round = max(1, budget // rounds)
    x_adv, used = x.clone(), torch.zeros_like(x, dtype=torch.bool)
    for _ in range(rounds):
        g = _loss_grad(model, x_adv, y)
        sal = g.abs().masked_fill(used, -1.0).flatten(1)
        idx = sal.topk(per_round, dim=1).indices
        flat, uflat = x_adv.flatten(1).clone(), used.flatten(1).clone()
        flat.scatter_(1, idx, torch.sign(g.flatten(1).gather(1, idx)))
        uflat.scatter_(1, idx, True)
        x_adv, used = flat.view_as(x), uflat.view_as(x)
    return x_adv.detach()


def attack_deepfool(model, x, y, eps, max_iter=20, overshoot=0.02, num_classes=10):
    """Batched multi-class DeepFool (minimal L2 step to the nearest boundary), ε-clipped."""
    x0, b = x.clone().detach(), len(x)
    ar = torch.arange(b)
    r_tot = torch.zeros_like(x0)
    for _ in range(max_iter):
        x_adv = torch.clamp(x0 + (1 + overshoot) * r_tot, -1, 1).requires_grad_(True)
        logits = model(x_adv)
        active = logits.argmax(1) == y
        if not active.any():
            break
        grads = torch.stack([torch.autograd.grad(logits[:, k].sum(), x_adv, retain_graph=True)[0]
                             for k in range(num_classes)], 1)
        f = logits.detach()
        w = grads - grads[ar, y][:, None]
        fdiff = (f - f.gather(1, y[:, None])).abs()
        wnorm = w.flatten(2).norm(dim=2) + 1e-8
        ratio = fdiff / wnorm
        ratio[ar, y] = float("inf")
        l = ratio.argmin(1)
        r_i = ((ratio[ar, l] + 1e-4) / wnorm[ar, l])[:, None, None, None] * w[ar, l]
        r_tot = r_tot + r_i * active[:, None, None, None]
    return _clip_ball(torch.clamp(x0 + (1 + overshoot) * r_tot, -1, 1), x0, eps).detach()


def attack_spatial(model, x, y, eps):
    """Worst-case rotation + translation from a grid (max shift ≈ 15ε px, rotation ≈ 100ε°)."""
    shift, rot = min(3, max(1, int(round(eps * 15)))), eps * 100
    best, best_loss = x.clone(), torch.full((len(x),), -float("inf"), device=x.device)
    with torch.no_grad():
        for angle in (-rot, 0.0, rot):
            r = TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=[-1.0]) if angle else x
            for dx in range(-shift, shift + 1):
                for dy in range(-shift, shift + 1):
                    cand = torch.roll(r, shifts=(dx, dy), dims=(2, 3))
                    loss = F.cross_entropy(model(cand), y, reduction="none")
                    better = loss > best_loss
                    best[better], best_loss[better] = cand[better], loss[better]
    return best


def attack_ead(model, x, y, eps, iters=20):
    """Elastic-net (EAD-style): signed ascent + L1 soft-thresholding → sparse perturbation."""
    alpha = eps / 5
    beta = alpha * 0.5
    delta = torch.zeros_like(x)
    for _ in range(iters):
        delta = delta + alpha * _loss_grad(model, x + delta, y).sign()
        delta = torch.sign(delta) * torch.clamp(delta.abs() - beta, min=0)
        delta = _clip_ball(x + delta, x, eps) - x
    return (x + delta).detach()


def attack_patch(model, x, y, eps, steps=20):
    """Optimised square patch (side ≈ 4+25ε px) placed on the most salient region."""
    size = int(min(14, 4 + round(eps * 25)))
    sal = F.avg_pool2d(_loss_grad(model, x, y).abs(), size, stride=1)
    pos = sal.flatten(1).argmax(1)
    rows, cols = pos // sal.shape[-1], pos % sal.shape[-1]
    mask = torch.zeros_like(x)
    for i, (r, c) in enumerate(zip(rows.tolist(), cols.tolist())):
        mask[i, :, r:r + size, c:c + size] = 1
    x_adv = x * (1 - mask) + mask * (torch.rand_like(x) * 2 - 1)
    for _ in range(steps):
        x_adv = torch.clamp(x_adv + 0.25 * mask * _loss_grad(model, x_adv, y).sign(), -1, 1)
    return x_adv.detach()


def attack_blur(model, x, y, eps):
    """Non-adversarial baseline: Gaussian blur with σ = 5ε (common corruption)."""
    sigma = max(0.1, eps * 5)
    k = min(9, 2 * math.ceil(2 * sigma) + 1)
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])


def attack_feature(model, x, y, eps, iters=10):
    """Feature-space PGD: push the penultimate embedding away from the clean one."""
    with torch.no_grad():
        f0 = model.embed(x)
    x_adv = _clip_ball(x + torch.empty_like(x).uniform_(-eps / 2, eps / 2), x, eps)
    for _ in range(iters):
        xa = x_adv.clone().requires_grad_(True)
        g, = torch.autograd.grad(((model.embed(xa) - f0) ** 2).sum(), xa)
        x_adv = _clip_ball(x_adv + eps / 4 * g.sign(), x, eps)
    return x_adv.detach()


VISION_ATTACKS = {
    "11. FGSM (Fast Gradient Sign Method)": attack_fgsm,
    "12. PGD (Projected Gradient Descent)": attack_pgd,
    "13. C&W L2 (ε-bounded)": attack_cw,
    "14. JSMA Saliency Map (L0)": attack_jsma,
    "15. DeepFool (ε-clipped)": attack_deepfool,
    "16. Spatial Rotation + Shift": attack_spatial,
    "17. Elastic-Net (EAD sparse)": attack_ead,
    "18. Adversarial Patch": attack_patch,
    "19. Gaussian Blur (corruption baseline)": attack_blur,
    "20. Feature-Space Perturbation": attack_feature,
}


# =============================================================================
# VISION DEFENSES
# =============================================================================
VISION_DEFENSES = ["Bit-Depth Reduction", "Spatial Median Blur", "Randomized Smoothing"]


def bit_depth_reduction(x, bits=3):
    levels = 2 ** bits - 1
    return torch.round((x + 1) / 2 * levels) / levels * 2 - 1


def median_blur(x, k=3):
    p = k // 2
    patches = F.pad(x, (p, p, p, p), mode="reflect").unfold(2, k, 1).unfold(3, k, 1)
    return patches.contiguous().flatten(-2).median(-1).values


def defended_probs(model, x, defenses, bits=3, sigma=0.25, n_samples=16):
    """Class probabilities after the defense pipeline (vote shares under smoothing)."""
    defenses = defenses or []
    with torch.no_grad():
        if "Bit-Depth Reduction" in defenses:
            x = bit_depth_reduction(x, bits)
        if "Spatial Median Blur" in defenses:
            x = median_blur(x)
        if "Randomized Smoothing" in defenses:
            votes = torch.zeros(len(x), 10)
            for _ in range(n_samples):
                noisy = torch.clamp(x + sigma * torch.randn_like(x), -1, 1)
                votes += F.one_hot(model(noisy).argmax(1), 10).float().cpu()
            return votes / n_samples
        return torch.softmax(model(x), 1).cpu()


def preprocess_for_display(x, defenses, bits=3):
    """The image the classifier sees after deterministic preprocessing defenses."""
    if "Bit-Depth Reduction" in (defenses or []):
        x = bit_depth_reduction(x, bits)
    if "Spatial Median Blur" in (defenses or []):
        x = median_blur(x)
    return x


def evaluate_vision_attack(model, x, y, attack, eps, defenses, bits=3, sigma=0.25, seed=0, batch=128) -> dict:
    torch.manual_seed(seed)
    fn = VISION_ATTACKS[attack]
    adv_parts = [fn(model, x[i:i + batch], y[i:i + batch], eps) for i in range(0, len(x), batch)]
    x_adv = torch.cat(adv_parts)
    with torch.no_grad():
        p_clean, p_adv = model(x).argmax(1), model(x_adv).argmax(1)
    d_clean = defended_probs(model, x, defenses, bits, sigma).argmax(1)
    d_adv = defended_probs(model, x_adv, defenses, bits, sigma).argmax(1)
    correct = p_clean == y
    return {
        "clean_acc": correct.float().mean().item(),
        "adv_acc": (p_adv == y).float().mean().item(),
        "def_clean_acc": (d_clean == y).float().mean().item(),
        "def_adv_acc": (d_adv == y).float().mean().item(),
        "success_rate": ((p_adv != y) & correct).sum().item() / max(correct.sum().item(), 1),
        "linf": (x_adv - x).abs().flatten(1).max(1).values.mean().item(),
        "l2": (x_adv - x).flatten(1).norm(dim=1).mean().item(),
    }
