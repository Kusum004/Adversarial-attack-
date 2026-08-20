# 🛡️ Adversarial Machine Learning (AML) Security & Defense Lab

[![GitHub Repository](https://img.shields.io/badge/GitHub-Adversarial--attack--blue?logo=github)](https://github.com/Kusum004/Adversarial-attack-.git)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](http://localhost:8501)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)

An end-to-end interactive **Adversarial Machine Learning Security & Defense Evaluation Benchmark** built with Streamlit, PyTorch, Scikit-Learn, and Plotly. The platform evaluates **20 Adversarial Attacks** (10 Tabular, 10 Vision) and **20 Counter-Defense Mechanisms** across supervised classifiers, unsupervised clustering algorithms, and deep learning convolutional neural networks (CNNs).

---

## 📌 Repository & Quick Links

- **GitHub Repository**: [https://github.com/Kusum004/Adversarial-attack-.git](https://github.com/Kusum004/Adversarial-attack-.git)
- **Main Live Dashboard App**: [`app.py`](file:///c:/Users/S%20Kusum/Documents/AML/CIA_3/app.py)
- **Core Evaluation Notebook**: [`AML_CIA3_2548532.ipynb`](file:///c:/Users/S%20Kusum/Documents/AML/CIA_3/AML_CIA3_2548532.ipynb)
- **Local Application URL**: [http://localhost:8501](http://localhost:8501)
- **Public Streamlit Cloud Portal**: [share.streamlit.io](https://share.streamlit.io/)

---

## 📊 Dataset Architecture

### 1. Tabular E-Commerce Fraud Dataset
Simulated Kaggle E-Commerce Fraud Benchmark matching financial risk profiling:
- **Sample Count**: 2,000 transaction records (80% Train, 20% Test split).
- **Features (5)**: `transaction_amount`, `customer_age`, `account_age_days`, `quantity`, `transaction_hour`.
- **Scaling**: `StandardScaler` (Z-score normalization).
- **Target Label**: `is_fraud` (0: Legitimate, 1: Fraud — 15% Imbalanced Fraud ratio).

### 2. Vision E-Commerce Catalog Product Dataset
Fashion-MNIST Product Catalog Benchmark representing online store inventory:
- **Format**: $28 \times 28$ grayscale images normalized in range $[-1.0, 1.0]$.
- **Product Classes (10)**: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot.

---

## ⚙️ Model Architecture & Benchmark Suite

### Supervised Classifiers & Clusterers (Scikit-Learn)
1. **Logistic Regression**: Linear decision boundary classifier.
2. **Support Vector Machine (SVM)**: Radial basis function (RBF) margin classifier.
3. **Random Forest Classifier**: Ensemble decision tree classifier ($N=50$ estimators).
4. **Gradient Boosting Classifier**: Sequential tree boosting classifier.
5. **K-Means Clustering**: $K=2$ cluster centroid partitioner.
6. **DBSCAN Clustering**: Density-based spatial clustering ($\epsilon=0.5$, $\text{min\_samples}=5$).

### Deep Convolutional Neural Network (PyTorch)
- **Architecture (`SimpleCNN`)**:
  - `Conv2D(1 -> 16, kernel=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2D(2, 2)`
  - `Conv2D(16 -> 32, kernel=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2D(2, 2)`
  - `Linear(32 * 7 * 7 -> 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(128 -> 10)`

---

## ⚡ Adversarial Attacks (20 Attacks Total)

### Tabular Data Attacks (Attacks 1 – 10)
| Attack ID | Attack Name | Target Model | Perturbation Strategy |
| :--- | :--- | :--- | :--- |
| **Attack 1** | Feature Manipulation | Logistic Regression | Multiplicative scaling on `transaction_amount` ($X \times 3.5$) |
| **Attack 2** | Threshold Bypassing | Gradient Boosting | Minimal additive shift across tree node split boundaries |
| **Attack 3** | HopSkipJump Approx. | Random Forest | Boundary-seeking sign-guided gradient step ($+0.25$) |
| **Attack 4** | ZOO Finite Differences | Gradient Boosting | Zeroth-order optimization finite difference gradient estimator |
| **Attack 5** | NewtonFool Approx. | Logistic Regression | Newton-step adversarial iteration toward nearest decision boundary |
| **Attack 6** | Boundary Random Walk | SVM | Laplace-distributed stochastic perturbation ($\text{scale}=0.20$) |
| **Attack 7** | Uniform Noise Injection | K-Means & DBSCAN | Uniform random noise addition $U(-0.5, 0.5)$ |
| **Attack 8** | Centroid Displacement | K-Means | Vector shift pushing points toward neighboring cluster centroid |
| **Attack 9** | Core Density Modification | DBSCAN | High-variance Gaussian noise destroying dense core neighborhoods |
| **Attack 10** | Shadow Query Perturbation | Random Forest | Discrete binary perturbations along decision tree queries |

### Vision CNN Attacks (Attacks 11 – 20)
| Attack ID | Attack Name | Target Model | Perturbation Strategy |
| :--- | :--- | :--- | :--- |
| **Attack 11** | FGSM (Fast Gradient Sign) | PyTorch CNN | Single-step gradient sign perturbation: $x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L)$ |
| **Attack 12** | PGD (Projected Gradient Descent) | PyTorch CNN | Multi-step iterative gradient attack with $L_\infty$ ball projection |
| **Attack 13** | C&W $L_2$ Approximation | PyTorch CNN | Gaussian-bounded optimization targeting classifier margins |
| **Attack 14** | JSMA Saliency Map | PyTorch CNN | Saliency-masked perturbation targeting high-impact pixel regions |
| **Attack 15** | DeepFool Approximation | PyTorch CNN | Minimal vector projection onto linear decision hyperplanes |
| **Attack 16** | Spatial Shift Transformation | PyTorch CNN | Discrete affine pixel roll translation along spatial dimensions |
| **Attack 17** | Elastic Net (EAD Patch) | PyTorch CNN | Sparse $L_1/L_2$ combined localized perturbation |
| **Attack 18** | Localized Patch Attack | PyTorch CNN | Concentrated high-intensity adversarial square patch ($8 \times 8$) |
| **Attack 19** | Wasserstein Blur | PyTorch CNN | Wasserstein distance-bounded spatial blur perturbation |
| **Attack 20** | Feature Perturbation | PyTorch CNN | Dense feature map gaussian degradation |

---

## 🛡️ Counter-Defense Mechanisms (20 Defenses Total)

### Tabular Defenses (Defenses 1 – 10)
1. **Feature Squeezing / Quantization**: Binning continuous features into discrete steps to eliminate sub-threshold perturbations.
2. **Gaussian Noise Smoothing**: Adding small Gaussian noise to obscure precise adversarial directions.
3. **Thermometer Encoding**: One-hot bin discretization across normalized feature ranges.
4. **Isolation Forest Outlier Purging**: Flagging and purging out-of-distribution adversarial vectors ($\text{contamination}=0.1$).
5. **KNN Consensus Consistency**: $K$-Nearest Neighbor majority voting sanitization.
6. **Clipping Bounds**: Hard bounding feature ranges within valid feature scale $[-2.5, 2.5]$.
7. **Median Filter Sanitization**: Rolling spatial median filter across feature dimensions.
8. **Quantile Sanitization**: Interquartile range (IQR) outlier trimming ($Q_1 - 1.5 \times \text{IQR}$, $Q_3 + 1.5 \times \text{IQR}$).
9. **Ensemble Majority Voting**: Combining predictions across heterogeneous classifier families.
10. **Loss-Based Confidence Filter**: Filtering predictions falling below threshold prediction confidence.

### Vision Defenses (Defenses 11 – 20)
11. **Randomized Smoothing**: Injecting isotropic Gaussian noise during inference.
12. **Activation Clipping**: Constraining intermediate activation bounds inside $[-0.8, 0.8]$.
13. **Bit-Depth Quantization**: Reducing pixel bit-depth (e.g., 3-bit color depth) to purge high-frequency adversarial noise.
14. **Spatial Median Blur**: $3 \times 3$ local spatial median filter removing salt-and-pepper adversarial perturbations.
15. **JPEG Compression Simulation**: High-frequency Discrete Cosine Transform (DCT) quantization simulation.
16-20. Combined Defense Pipeline: Multi-stage feature squeezing + spatial median filter stack.

---

## 🖥️ Live Dashboard Architecture (`app.py`)

The dashboard is structured into 4 interactive workspace modules:

```
├── 📊 Tab 1: Dataset & EDA Explorer
│   ├── Raw E-Commerce Fraud Dataframe & Statistics
│   ├── Interactive Plotly Feature Correlation Heatmap
│   ├── Class Distribution & Imbalance Bar Charts
│   └── Fashion-MNIST Product Catalog Sample Gallery
│
├── 🛡️ Tab 2: Tabular AML Security Lab (Attacks 1-10)
│   ├── Model Selection (6 Algorithms) & Attack Dropdowns
│   ├── Interactive Perturbation Scale Slider (Epsilon)
│   ├── Live 2D PCA Space Boundary & Flipped Vector Plots
│   └── Metrics Evaluation Summary Table
│
├── 👁️ Tab 3: Vision CNN Security Lab (Attacks 11-20)
│   ├── Product Item Selector & Attack Controls
│   ├── 3-Stage Visual Pipeline: Clean ➔ Attacked ➔ Defended
│   └── 2x5 Multi-Attack Grid (Attacks 11-20 on Single Catalog Item)
│
└── 📈 Tab 4: Global Benchmark & Real-time Stress Simulator
    ├── Comprehensive 20-Attack Accuracy Recovery Overview
    └── Real-time Fraud Transaction Risk Score Simulator
```

---

## 🚀 Installation & Local Execution Guide

### Prerequisites
- Python 3.10+
- Git

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/Kusum004/Adversarial-attack-.git
cd Adversarial-attack-

pip install -r requirements.txt
```

### 2. Launch Streamlit Dashboard
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser!

---

## 🌐 Public Cloud Deployment (Streamlit Community Cloud)

1. Fork or push to your GitHub account: `https://github.com/Kusum004/Adversarial-attack-.git`.
2. Navigate to **[share.streamlit.io](https://share.streamlit.io/)**.
3. Click **Deploy an app** and configure:
   - **Repository**: `Kusum004/Adversarial-attack-`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!**

---

## 📄 License & Course Information

- **Course**: Advanced Machine Learning (AML) Benchmark Evaluation CIA-3
- **Author / Repository**: [Kusum004/Adversarial-attack-](https://github.com/Kusum004/Adversarial-attack-.git)
