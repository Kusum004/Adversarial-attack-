# 🛡️ Adversarial Machine Learning (AML) Security & Defense Lab

[![GitHub Repository](https://img.shields.io/badge/GitHub-Adversarial--attack--blue?logo=github)](https://github.com/Kusum004/Adversarial-attack-.git)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](http://localhost:8501)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)

An interactive **Adversarial Machine Learning security & defense benchmark** built with Streamlit, PyTorch, scikit-learn and Plotly. It runs **20 adversarial attacks** (10 tabular, 10 vision) and **7 defenses** (4 tabular, 3 vision) against supervised classifiers, clustering algorithms and a trained convolutional neural network. **Every metric in the dashboard is computed live** from the selected models, attacks, budgets and defenses.

---

## 📌 Quick Links

- **GitHub Repository**: [Kusum004/Adversarial-attack-](https://github.com/Kusum004/Adversarial-attack-.git)
- **Dashboard**: [`app.py`](app.py) (UI) · [`aml_core.py`](aml_core.py) (data, models, attacks, defenses, evaluation)
- **CNN training script**: [`train_cnn.py`](train_cnn.py) → saves [`models/fashion_cnn.pt`](models/)
- **Notebook**: [`AML_CIA3_2548532.ipynb`](AML_CIA3_2548532.ipynb)
- **Tests**: [`tests/test_core.py`](tests/test_core.py)

---

## 🎯 Threat Model

- **Tabular (fraud detection)**: the attacker is a fraudster who perturbs the features of *fraudulent* transactions so they get approved. The perturbation is bounded in L∞ by ε (in standardized feature units) and clipped to the feasible feature range seen in training data. Legitimate transactions are untouched.
- **Clustering (K-Means / DBSCAN)**: untargeted perturbations that try to change cluster structure.
- **Vision (catalog classification)**: untargeted misclassification in normalized [-1, 1] pixel space, L∞-bounded by ε unless an attack states otherwise (JSMA = L0, patch = physical patch, spatial = geometric).
- **Defenses** are evaluated against attacks crafted on the *undefended* model (an oblivious adversary). An adaptive attacker who knows the defense would do better, so defended numbers are an upper bound on real robustness.

---

## 📊 Datasets

### 1. Tabular E-Commerce Fraud Dataset (synthetic)
- **Samples**: 2,000 transactions, stratified 80/20 train/test split. Seed-controlled from the sidebar.
- **Features (5)**: `transaction_amount`, `customer_age`, `account_age_days`, `quantity`, `transaction_hour`.
- **Label**: `is_fraud`, exactly 15% fraud. Fraud risk rises with large amounts, new accounts, night-time purchases (23:00–05:00), large quantities and slightly younger customers, plus Gaussian label noise. The models therefore have a real but imperfect signal to learn.
- **Scaling**: `StandardScaler` fitted on the training split only.

### 2. Vision Product Catalog Dataset (Fashion-MNIST)
- 28×28 grayscale images normalized to [-1, 1]; 10 classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot.
- Downloaded automatically to `data/` (git-ignored).

---

## ⚙️ Models

| Model | Details | Clean test result (seed 42) |
| :--- | :--- | :--- |
| Logistic Regression | `max_iter=1000` | 94.0% acc · 71.7% fraud recall |
| SVM | RBF kernel | 93.0% acc · 66.7% fraud recall |
| Random Forest | 100 trees | 94.3% acc · 76.7% fraud recall |
| Gradient Boosting | sklearn defaults | 92.7% acc · 70.0% fraud recall |
| K-Means | K = 2, `n_init=10` | silhouette ≈ 0.155 |
| DBSCAN | ε = 1.0, `min_samples=5` | ≈ 79% non-noise points |
| **SimpleCNN** (PyTorch) | Conv(1→16)→ReLU→Pool→Conv(16→32)→ReLU→Pool→FC(1568→128)→ReLU→FC(128→10); Adam, 4 epochs | **89.5% test accuracy** |

A model that always answers "legitimate" scores 85% accuracy, so fraud recall is the more informative tabular metric.

---

## ⚡ Adversarial Attacks (20)

### Tabular Attacks (1 – 10)
| # | Attack | Default target | Knowledge | How it works |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Feature Manipulation / FGSM | Logistic Regression | White-box | Moves every feature by ε against sign(w) — exact FGSM for a linear model |
| 2 | Threshold Bypassing | Gradient Boosting | Score-based | Greedy coordinate search over {−ε, 0, +ε} per feature to slip under split thresholds |
| 3 | HopSkipJump Boundary Search | Random Forest | Decision-only | Heads toward the nearest training point the model calls legitimate, binary-searches the smallest flipping step |
| 4 | ZOO Finite Differences | Gradient Boosting | Score-based | Finite-difference gradient estimate + 10 signed descent steps |
| 5 | NewtonFool | Logistic Regression | White-box | Newton step on the logit to the decision boundary (minimal L2), clipped to ε |
| 6 | Boundary Random Walk | SVM | Score-based | 30-step random walk accepting moves that lower the fraud score |
| 7 | Uniform Random Noise | K-Means | None | U(−ε, ε) noise on every feature |
| 8 | Centroid Displacement | K-Means | White-box | Pushes each point toward the nearest *other* centroid |
| 9 | Core-Point Density Modification | DBSCAN | None | Gaussian jitter (σ = ε) that breaks dense core neighborhoods |
| 10 | Shadow-Model Transfer | Random Forest | Black-box | Queries the target, trains a surrogate LR on its answers, FGSM on the surrogate |

Attacks 2, 3, 4, 6 and 10 only need query access, so the dashboard lets you point them at any classifier.

### Vision Attacks (11 – 20)
| # | Attack | Norm | How it works |
| :--- | :--- | :--- | :--- |
| 11 | FGSM | L∞ | x + ε·sign(∇ₓL) |
| 12 | PGD | L∞ | Random start, 10 steps of size ε/4, projected to the ε-ball |
| 13 | C&W L2 | L2 objective, ε-bounded | Adam on tanh-space variables minimizing ‖δ‖² + c·max(Z_y − max Z_{i≠y}, 0) |
| 14 | JSMA Saliency Map | L0 | Saturates the ≈ ε·200 most salient pixels over 5 rounds |
| 15 | DeepFool | L2 → clipped to ε | Batched multi-class DeepFool (linearized minimal step to the nearest boundary) |
| 16 | Spatial Rotation + Shift | Geometric | Worst case over a grid of rotations (±100ε°) and translations (up to 3 px) |
| 17 | Elastic-Net (EAD) | L∞ + L1 | Signed ascent with L1 soft-thresholding → sparse perturbation |
| 18 | Adversarial Patch | Patch | Square patch (≈ 4 + 25ε px) on the most salient region, optimized for 20 steps |
| 19 | Gaussian Blur | Corruption | Non-adversarial baseline, σ = 5ε |
| 20 | Feature-Space Perturbation | L∞ | PGD that maximizes the distance of penultimate-layer features from the clean ones |

---

## 🛡️ Defenses (7)

### Tabular
1. **Feature Squeezing & Clipping Bounds**: clip to the training range, then quantize to a 0.25 grid.
2. **Isolation Forest Outlier Purging**: transactions flagged as anomalous (contamination 5%) are escalated as fraud instead of auto-approved.
3. **KNN Neighborhood Consensus**: replace the input with the mean of its 7 nearest training points before classifying.
4. **Quantile Sanitization**: clip each feature to the training 2.5th–97.5th percentiles.

### Vision
5. **Bit-Depth Reduction**: quantize pixels to 2^bits levels (1–8 bits).
6. **Spatial Median Blur**: 3×3 median filter.
7. **Randomized Smoothing**: majority vote over 16 Gaussian-noised copies (σ adjustable).

Vision defenses can be stacked in any combination.

---

## 📈 Sample Results (seed 42, computed by the dashboard)

**Tabular, ε = 0.35, no defense:** every evasion attack cuts fraud recall from ≈ 0.67–0.77 down to 0.25–0.40 (43–67% of detected fraud evades). Feature squeezing gives the largest recovery against query-based attacks (HopSkipJump on RF: 0.40 → 0.62 recall).

**Vision, ε = 0.15, 256 test images (clean accuracy 91.4%):**

| Attack | Attacked acc. | + Bit-depth(3) & Median | + Randomized smoothing (σ = 0.25) |
| :--- | :---: | :---: | :---: |
| FGSM | 25.8% | 14.8% | 40.2% |
| PGD | 9.0% | 9.4% | 30.1% |
| C&W L2 | 44.9% | 77.3% | 62.9% |
| JSMA | 16.4% | 51.6% | 16.4% |
| DeepFool | 73.4% | 82.4% | 71.9% |
| Adversarial Patch | 9.0% | 23.0% | 10.2% |

No single defense works against everything. Input-transformation defenses also cost clean accuracy (91.4% → 87.1% with bit-depth + median).

---

## 🖥️ Dashboard Layout

```
├── KPI row — live baseline, attacked and defended metrics for RF and the CNN
├── 📊 Tab 1: Dataset & EDA — raw data, statistics, correlation, class balance,
│              per-class feature distributions, catalog gallery with CNN predictions
├── 🛡️ Tab 2: Tabular Lab — attack / target / ε / defense controls, PCA view with
│              evaded points highlighted, metric cards, all-10-attacks table
├── 👁️ Tab 3: Vision Lab — clean ➔ attacked ➔ defended ➔ perturbation panels,
│              class probabilities, batch metrics, 10-attack grid and table
└── 📈 Tab 4: Benchmark — clean/attacked/defended bars, robustness curves (recall
               or accuracy vs ε), fraud stress simulator per model
```

---

## 🚀 Running Locally

```bash
git clone https://github.com/Kusum004/Adversarial-attack-.git
cd Adversarial-attack-
pip install -r requirements.txt

streamlit run app.py          # open http://localhost:8501
```

The trained CNN weights (`models/fashion_cnn.pt`, ~0.8 MB) are included. To retrain:

```bash
python train_cnn.py --epochs 4
```

If the weights file is missing, the dashboard trains the CNN on first load (about a minute on CPU).

Run the tests:

```bash
pip install pytest
python -m pytest -q
```

---

## 🌐 Deploying to Streamlit Community Cloud

1. Push the repository to GitHub.
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** → **Deploy an app**.
3. Repository `Kusum004/Adversarial-attack-`, branch `main`, main file `app.py`.

---

## 📄 Course Information

- **Course**: Advanced Machine Learning (AML) — CIA-3
- **Repository**: [Kusum004/Adversarial-attack-](https://github.com/Kusum004/Adversarial-attack-.git)
