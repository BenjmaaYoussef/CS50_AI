# DataCo Smart Supply Chain — ML Project Documentation (v3)

## CS280/CS485 Introduction to Artificial Intelligence — Spring 2026

---

## 1. Project Overview

This project applies supervised machine learning to a real-world supply chain problem: predicting whether an order will be delivered late **before it ships**. The goal is to give logistics teams an early-warning system so they can proactively reroute, prioritize, or communicate delays. The project covers the complete ML pipeline — preprocessing, modelling with five architecturally diverse classifiers, hyperparameter tuning, scientific evaluation, and inference on unseen data.

**Domain:** Business / Marketing / Supply Chain (permitted domain per course guidelines)  
**Learning Paradigm:** Supervised Classification (binary)  
**Target Variable:** `Late_delivery_risk` — 1 = late, 0 = on time

---

## 2. Dataset Description

### 2.1 Source and License

| Attribute     | Detail                                                                                       |
| :------------ | :------------------------------------------------------------------------------------------- |
| Name          | DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS                                              |
| Publisher     | DataCo Global (published on Mendeley Data)                                                   |
| Mendeley DOI  | https://data.mendeley.com/datasets/8gx2fvg2k6/5                                              |
| Kaggle mirror | https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis |
| License       | Creative Commons Attribution 4.0 (CC BY 4.0) — free to use with attribution                  |
| File          | `DataCoSupplyChainDataset.csv`                                                               |
| Rows          | 180,520                                                                                      |
| Columns       | 53                                                                                           |

### 2.2 Dataset Structure

Each row represents one customer order. The 53 columns encode three intersecting dimensions simultaneously — **marketing**, **logistics**, and **financial** — making the dataset naturally suited to this project's domain without requiring artificial construction.

| Dimension         | Example Columns                                                                           |
| :---------------- | :---------------------------------------------------------------------------------------- |
| Customer & Market | Customer Segment (Consumer/Corporate/Home Office), Market, Customer Country, Order Region |
| Product           | Category Name, Product Name, Product Price, Order Item Quantity                           |
| Financial         | Order Item Discount Rate, Order Item Profit Ratio, Sales, Order Profit Per Order          |
| Logistics         | Shipping Mode, Days for Shipping (Real), Days for Shipment (Scheduled), Delivery Status   |
| Target            | `Late_delivery_risk` (binary: 0 = on time, 1 = late)                                      |

### 2.3 Why This Dataset is Well-Suited

- **Scale:** 180,520 records gives all five models sufficient training data and makes statistical evaluation reliable.
- **Native domain intersection:** Marketing features (segment, discount, category) and supply chain features (shipping mode, delivery days) coexist in every row — the intersection is structural, not forced.
- **Real-world imbalance:** The class distribution of `Late_delivery_risk` should be checked. If it is approximately 55–60% late vs. 40–45% on time, accuracy alone becomes misleading — which motivates the inclusion of F1-score, Precision-Recall curves, and ROC-AUC as primary metrics.
- **Open license:** CC BY 4.0 allows academic use and publication without restriction.

---

## 3. Problem Definition

### 3.1 Business Objective

Identify orders at risk of late delivery **at the time the order is placed**, before shipment begins, so logistics teams can intervene (e.g., prioritize packing, upgrade shipping mode, notify customer).

### 3.2 ML Formulation

> **Binary Classification:** Given the features known at order time, predict `Late_delivery_risk ∈ {0, 1}`.

The constraint "order-time features only" is critical. Any column that reveals information only available _after_ shipment begins must be removed to avoid data leakage.

---

## 4. Preprocessing Pipeline

> **Notebook structure note:** Every preprocessing step below must have a Markdown cell _above_ (explaining the decision) and an interpretation cell _below_ every output. The notebook is the presentation artifact — it must read as a self-contained scientific document from top to bottom.

### 4.0 Inference Holdout (FIRST STEP — Before Any Preprocessing)

Before anything else, hold out 10 rows for the final inference stage:

```python
df_inference = df.sample(10, random_state=42)
df = df.drop(df_inference.index).reset_index(drop=True)
```

This guarantees these rows never touch any preprocessing object (scaler, encoder) during fitting, and eliminates any ambiguity about leakage from the test set.

### 4.1 Data Quality Audit & Noise Handling

Before any transformation, audit data quality across all 53 columns. Document: initial shape, missing value distribution, duplicate count, target class balance, and per-feature outlier statistics. This audit is a graded transparency component.

#### 4.1.1 Missing Value Strategy

- Columns with >40–50% missing values → **drop entirely**
- Rows with sparse, random missingness (<1–2% of dataset) → **drop rows**
- Numerical columns with moderate missingness → **median imputation** (robust to outliers)
- Categorical columns → **mode imputation**

Every decision is documented with justification in the notebook's markdown cells.

#### 4.1.2 Duplicate Detection

Identify and remove exact duplicate rows. Log the count of duplicates removed.

#### 4.1.3 Outlier Management

Audit key numerical columns using the IQR method. Treatment:

- Cap extreme values at the 1st and 99th percentiles (winsorization) to preserve row count
- Use `RobustScaler` (median + IQR) instead of `StandardScaler` for outlier-sensitive models
- Remove rows only when values are clearly erroneous (e.g., negative prices, discount > 100%)

#### 4.1.4 Data Type Validation

Parse date fields as `datetime`, ensure numerical values are not stored as strings, encode binary flags as integers. Log all type corrections.

#### 4.1.5 Feature Selection & Dimensionality Control

After encoding, apply two lightweight filters:

- **Near-zero variance filter:** drop columns where almost all values are identical
- **Correlation filter:** drop one of any pair of numerical features with correlation > 0.95

These reduce noise for k-NN and dimensionality risk across all models.

### 4.2 Data Leakage Removal (CRITICAL)

Remove any column that reveals information available only _after_ the order has shipped or delivered. Includes columns that directly encode the target or post-shipment outcomes. Each dropped column is justified with reference to the business constraint: predictions must be made at order time.

### 4.3 Identifier Removal: Programmatic Strategy

Apply a rule-based strategy — not a hard-coded column list:

1. **Uniqueness threshold:** drop columns where >95% of values are unique (order IDs, customer IDs)
2. **Name-based filtering:** drop columns whose names contain: `id`, `email`, `password`, `image`, `url`, `fname`, `lname`, `phone`
3. **Pattern-based detection:** detect and drop fields containing email formats, URL structures, or file paths
4. **Cardinality review:** group categories appearing in <2 rows into an "Other" bucket or drop

All dropped columns are logged with the rule that triggered removal.

### 4.4 Feature Encoding

Use One-Hot Encoding for nominal categorical variables (Shipping Mode, Customer Segment, Order Region, Category Name, Market). No ordinal encoding for unordered categories. Monitor resulting column count; group low-frequency categories into "Other" if dimensionality becomes unwieldy.

### 4.5 Numerical Feature Scaling

Use `RobustScaler` (based on median and IQR) rather than `StandardScaler`, given the outlier management context. This is essential for:

- **k-NN:** Euclidean distance is distorted by different feature scales
- **SVM:** margin maximization is biased toward high-magnitude features
- **Perceptron/SGDClassifier:** weight updates are proportional to feature magnitude

Tree-based models are scale-invariant, but a single unified pipeline is applied for reproducibility. **The scaler is fit only on the training set and applied to train + test — never refit on the test set.**

### 4.6 Train-Test Split

Stratified 80/20 split **after** the 10-row inference holdout. Stratification preserves class distribution in both subsets. Random state is fixed for reproducibility.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

---

## 5. Models

Five supervised classifiers spanning fundamentally distinct algorithmic families, ensuring scientifically meaningful comparison rather than minor variations of the same technique.

> **Neural network constraint (per guidelines):** This course is NOT a deep learning project. If a neural network is included, it must be a tiny NN with no more than 3 layers. Our five models do not include a neural network — the SGDClassifier is a single-layer linear classifier (Perceptron variant), which is well within limits. This distinction must be clear if questioned.

---

### 5.1 k-Nearest Neighbors (k-NN)

**Core Idea:** Non-parametric, instance-based learner. Classifies a new point by finding the _k_ closest training examples by Euclidean distance and assigning the majority class. No explicit training phase — the entire training set is stored and queried at inference time.

**Mathematical Intuition:**

Given a query point **x**, find $\mathcal{N}_k(\mathbf{x})$ of the _k_ nearest training points:

$$d(\mathbf{x}, \mathbf{x_i}) = \sqrt{\sum_{j=1}^{n}(x_j - x_{ij})^2}$$

Predict the majority class:

$$\hat{y} = \arg\max_{c} \sum_{\mathbf{x_i} \in \mathcal{N}_k(\mathbf{x})} \mathbf{1}[y_i = c]$$

**Scalability Fix (CRITICAL):**  
k-NN has O(n·d) inference complexity per query. On 180K rows, this will cause unacceptable runtime before the presentation deadline. **Fix:** subsample the dataset to ~25,000 rows _exclusively for k-NN training_, using a stratified sample to preserve class balance:

```python
from sklearn.utils import resample
X_train_knn, y_train_knn = resample(
    X_train, y_train, n_samples=25000, stratify=y_train,
    replace=False, random_state=42  # replace=False: subsample WITHOUT replacement
)
```

> **BUG PREVENTION:** `resample()` defaults to `replace=True` (bootstrap). For subsampling, you MUST set `replace=False` to avoid duplicate rows in the training set.

This is scientifically valid and must be documented in a markdown cell. Additionally, set `algorithm='ball_tree'` and `n_jobs=-1` to maximize efficiency on even the subsampled set.

**Hyperparameters (GridSearchCV):**

| Hyperparameter    | Search Range               | Effect                                                                   |
| :---------------- | :------------------------- | :----------------------------------------------------------------------- |
| `n_neighbors` (k) | {3, 5, 7, 11, 15}          | Low k → complex boundary (overfitting); high k → smoother (underfitting) |
| `metric`          | {'euclidean', 'manhattan'} | Changes distance definition                                              |
| `weights`         | {'uniform', 'distance'}    | Distance-weighted voting gives closer neighbors more influence           |

**Connection to lecture:** Directly corresponds to k-nearest-neighbor classification in Lecture 4.

---

### 5.2 Perceptron / SGDClassifier (Modified Huber)

**Core Idea:** A single-layer linear classifier that learns a weight vector **w** such that the decision boundary is the hyperplane $\mathbf{w} \cdot \mathbf{x} = 0$.

**API Fix (CRITICAL):**  
`sklearn.linear_model.Perceptron` does **not** support `predict_proba()`, which is required for the inference stage's risk probability output and for plotting ROC/PR curves. **Fix:** use `SGDClassifier(loss='modified_huber')` instead. This implements the Perceptron learning rule with a soft threshold, supports `predict_proba()`, and — crucially — directly maps to the **soft threshold** introduced in Lecture 4, which the standard Perceptron does not.

**Lecture connection — Hard vs. Soft Threshold:**

The hard threshold (standard Perceptron, Lecture 4):

$$h_{\mathbf{w}}(\mathbf{x}) = \begin{cases} 1 & \text{if } \mathbf{w} \cdot \mathbf{x} \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

is non-differentiable and cannot produce probabilities. The soft threshold (Lecture 4) replaces the step function with a smooth sigmoid-like approximation, enabling probability output. `SGDClassifier(loss='modified_huber')` is the scikit-learn implementation of this concept.

Weight update rule (Perceptron Learning Rule):

$$w_i \leftarrow w_i + \alpha(y - h_{\mathbf{w}}(\mathbf{x})) \cdot x_i$$

**Hyperparameters (GridSearchCV):**

| Hyperparameter           | Search Range               | Effect                       |
| :----------------------- | :------------------------- | :--------------------------- |
| `alpha` (regularization) | {0.0001, 0.001, 0.01}      | L2 penalty on weights — THIS IS lambda in the lecture formula cost(h) = loss(h) + lambda*complexity(h) |
| `learning_rate`          | {'constant'}               | MUST be 'constant' for eta0 to take effect (default 'optimal' ignores eta0) |
| `max_iter`               | {500, 1000}                | Maximum epochs               |
| `eta0` (learning rate)   | {0.001, 0.01, 0.1}         | Step size for weight updates (alpha in the perceptron learning rule from Lecture 4) |

**Expected behavior:** Moderate accuracy. The supply chain task is not linearly separable, so the performance gap between this model and the ensemble methods quantifies the benefit of non-linear decision boundaries — the key scientific insight this model contributes.

---

### 5.3 Support Vector Machine (SVM)

**Core Idea:** Finds the hyperplane that maximizes the margin — the distance between the boundary and the nearest training points of each class (support vectors).

**Scalability Fix (CRITICAL):**  
`SVC(kernel='rbf')` has O(n^2) to O(n^3) training complexity. On 180,520 rows, this will either take hours or fail to complete before the presentation deadline. **Fix:** use `LinearSVC` for the linear case. Apply the same ~25,000-row subsample as k-NN if an RBF kernel is required, with explicit markdown justification. The non-linearity argument is already covered by the two ensemble models.

**Mathematical Intuition:**

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{subject to } y_i(\mathbf{w} \cdot \mathbf{x_i} + b) \geq 1$$

The $\frac{1}{2}\|\mathbf{w}\|^2$ term is L2 regularization — directly connecting to Lecture 4's regularization framework:

$$\text{cost}(h) = \text{loss}(h) + \lambda \cdot \text{complexity}(h)$$

**Hyperparameters (GridSearchCV):**

| Hyperparameter | Search Range       | Effect                                                 |
| :------------- | :----------------- | :----------------------------------------------------- |
| `C`            | {0.01, 0.1, 1, 10} | Low C → wide margin; high C → fewer misclassifications |
| `max_iter`     | {1000, 5000}       | Convergence budget                                     |

> If using `SVC` on a subsample: add `kernel: ['linear', 'rbf']` and `gamma: ['scale', 'auto']` to the grid.

**Note on `predict_proba()`:** `LinearSVC` does not natively support `predict_proba()`. Wrap with `CalibratedClassifierCV(LinearSVC(...))` to enable probability output for ROC/PR curves and inference.

> **BUG PREVENTION — GridSearchCV param names:** When wrapping LinearSVC inside CalibratedClassifierCV, hyperparameter names change. Use `estimator__C` (not `C`) in the param grid, e.g.:
> ```python
> param_grid = {'estimator__C': [0.01, 0.1, 1, 10], 'estimator__max_iter': [1000, 5000]}
> ```
> Otherwise GridSearchCV will crash with "Invalid parameter C".

**Connection to lecture:** Maximum margin separator from Lecture 4. L2 regularization connects explicitly to Lecture 4's regularization framework.

---

### 5.4 Random Forest _(additional model)_

**Core Idea:** Ensemble of decision trees trained on random subsets of data and features (bagging + feature randomness). Each tree votes; majority vote is the final prediction.

**Mathematical Intuition:**

Each tree $T_b$ is trained on a bootstrap sample $Z_b$. At each node, $m = \sqrt{p}$ random features are considered. Final prediction:

$$\hat{y} = \arg\max_{c} \sum_{b=1}^{B} \mathbf{1}[T_b(\mathbf{x}) = c]$$

**Hyperparameters (RandomizedSearchCV, n_iter=20):**

| Hyperparameter     | Search Range       | Effect                                        |
| :----------------- | :----------------- | :-------------------------------------------- |
| `n_estimators`     | {100, 200, 500}    | More trees → lower variance, higher compute   |
| `max_depth`        | {None, 10, 20, 30} | Individual tree complexity                    |
| `max_features`     | {'sqrt', 'log2'}   | Features per split                            |
| `min_samples_leaf` | {1, 5, 10}         | Minimum leaf size; higher = smoother boundary |

> **Why RandomizedSearchCV?** The full cross-product for RF and XGBoost would require thousands of model fits on 144K rows. `RandomizedSearchCV(n_iter=20, cv=5)` samples 20 random combinations — scientifically equivalent for finding good hyperparameters while being tractable. This must be documented in a markdown cell.

**Advantages:**

- Scale-invariant (no scaler required, but unified pipeline applies it for consistency)
- Provides **feature importances** — directly interpretable for the business objective
- Resistant to overfitting via bagging

---

### 5.5 XGBoost _(additional model)_

**Core Idea:** Sequential ensemble (boosting) where each new tree corrects the residual errors of the previous ensemble via gradient descent on the loss.

**Mathematical Intuition:**

At each round _t_, a new tree $f_t$ minimizes:

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} \ell(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x_i})) + \Omega(f_t)$$

where $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda\|\mathbf{w}\|^2$ penalizes tree complexity. This is **regularized boosting** — directly connecting to Lecture 4's regularization vocabulary.

**Hyperparameters (RandomizedSearchCV, n_iter=20):**

| Hyperparameter      | Search Range     | Effect                               |
| :------------------ | :--------------- | :----------------------------------- |
| `n_estimators`      | {100, 300, 500}  | Boosting rounds                      |
| `learning_rate` (η) | {0.01, 0.1, 0.3} | Shrinks each tree's contribution     |
| `max_depth`         | {3, 5, 7}        | Tree depth; lower = less overfitting |
| `subsample`         | {0.7, 0.9, 1.0}  | Row fraction per tree                |
| `colsample_bytree`  | {0.7, 0.9, 1.0}  | Feature fraction per tree            |

**Advantages:**

- Efficient on 180K rows via parallel computation (no subsampling needed)
- Built-in L1/L2 regularization
- `scale_pos_weight` for class imbalance handling
- Typically top performer on tabular data benchmarks

---

## 6. Model Comparison Reference

| Property               | k-NN               | SGDClassifier (Perceptron) | SVM (LinearSVC)                 | Random Forest      | XGBoost             |
| :--------------------- | :----------------- | :------------------------- | :------------------------------ | :----------------- | :------------------ |
| Algorithmic family     | Instance-based     | Linear (online)            | Margin-based linear             | Ensemble (bagging) | Ensemble (boosting) |
| Decision boundary      | Non-linear (local) | Linear                     | Linear                          | Non-linear         | Non-linear          |
| Requires scaling       | Yes                | Yes                        | Yes                             | No                 | No                  |
| Handles high dims      | Poor               | Good                       | Good                            | Good               | Good                |
| Supports predict_proba | Yes (via knn)      | Yes (modified_huber)       | Yes (via CalibratedClassifierCV)| Yes                | Yes                 |
| Scalability fix needed | Subsample 25K      | None                       | LinearSVC or subsample          | None               | None                |
| Expected performance   | Low-Medium         | Low                        | High                            | High               | Highest             |
| Lecture 4 coverage     | Direct             | Direct                     | Direct                          | Indirect           | Indirect            |

---

## 7. Evaluation Strategy

### 7.1 Metrics

All five models evaluated on the **same held-out test set** (20% stratified split):

| Metric    | Formula                                 | Why it matters here                                 |
| :-------- | :-------------------------------------- | :-------------------------------------------------- |
| Accuracy  | (TP+TN)/(TP+TN+FP+FN)                   | Baseline; potentially misleading if imbalanced      |
| Precision | TP/(TP+FP)                              | Of predicted late orders, how many were truly late? |
| Recall    | TP/(TP+FN)                              | Of truly late orders, how many did we catch?        |
| F1-Score  | 2·(Precision·Recall)/(Precision+Recall) | Balances precision and recall                       |
| ROC-AUC   | Area under ROC curve                    | Threshold-independent ranking quality               |

**Business context:** Missing a late delivery (false negative) is more costly than a false alarm. Recall for the "late" class is weighted highly in final model selection.

### 7.2 Lecture Connection — Loss Functions

The evaluation metrics connect directly to Lecture 4's loss function framework:
- **Accuracy** = 1 - (0-1 loss averaged over test set). The 0-1 loss function from Lecture 4: L(actual, predicted) = 0 if actual = predicted, 1 otherwise
- **F1/Precision/Recall** extend beyond 0-1 loss to distinguish between error types (FP vs FN), which the basic 0-1 loss cannot do
- **Regularization** (Lecture 4): cost(h) = loss(h) + lambda * complexity(h) — directly implemented via SGDClassifier's `alpha`, SVM's `C` (inverse of lambda), and XGBoost's `gamma`/`lambda`

This connection must be stated in the notebook and report.

### 7.3 Mandatory Figures

**A. Preprocessing Charts (in notebook sections 2-5):**
1. **Class Distribution Bar Chart** — Late vs On-Time counts with percentages
2. **Missing Values Bar Chart** — per-column missing count (top 15 columns)
3. **Correlation Heatmap** — numerical features, annotated, used to justify dropping correlated features
4. **Outlier Box Plots** — for key numerical features (price, quantity, discount, etc.)

**B. Model Evaluation Charts (in notebook sections 7-12):**
5. **Confusion Matrix** — one per model (5 total) — mandatory per guidelines
6. **ROC Curve Overlay** — all five models on one plot, AUC in legend
7. **Precision-Recall Curve** — especially important if class imbalance is detected
8. **k vs. F1 curve** — for k-NN (validation F1 across k values)
9. **n_estimators vs. F1 curve** — for Random Forest
10. **Feature Importance Bar Chart** — from Random Forest and/or XGBoost (top 10-15 features), with markdown interpretation tying features back to the business problem
11. **Train vs Test F1 Bar Chart** — all 5 models side-by-side, train F1 vs test F1. If train >> test, that's **overfitting** (Lecture 4). This chart directly demonstrates a lecture concept
12. **Cross-Validation Box Plot** — 5-fold CV score distributions per model, showing score stability

Each figure must be followed by a markdown interpretation cell explaining what the plot shows and what it implies for model selection.

### 7.3 Cross-Validation

**5-Fold Stratified Cross-Validation** on the training set before final test evaluation:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
```

### 7.4 Hyperparameter Tuning

- **k-NN, Perceptron, SVM:** `GridSearchCV(cv=5, scoring='f1')`
- **Random Forest, XGBoost:** `RandomizedSearchCV(n_iter=20, cv=5, scoring='f1', random_state=42)`

The best estimator is retrained on the full training set and evaluated **once** on the test set.

### 7.5 Comparison Table (Mandatory Deliverable)

A single summary table comparing all five models side-by-side on the test set:

| Model         | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :------------ | :------- | :-------- | :----- | :------- | :------ |
| k-NN          | ...      | ...       | ...    | ...      | ...     |
| SGDClassifier | ...      | ...       | ...    | ...      | ...     |
| LinearSVC     | ...      | ...       | ...    | ...      | ...     |
| Random Forest | ...      | ...       | ...    | ...      | ...     |
| XGBoost       | ...      | ...       | ...    | ...      | ...     |

This table must appear in both the notebook and the report, followed by interpretation.

---

## 8. Inference Stage

### 8.1 Procedure

The 10-row `df_inference` set was held out at the very start (Section 4.0), before any preprocessing object was fit.

1. Apply the same preprocessing pipeline: same scaler, same encoder — **no refitting**
2. Run `.predict()` for binary class label
3. Run `.predict_proba()[:, 1]` for late-delivery probability
4. Display table: Order features | True Label | Predicted Label | Risk Probability

### 8.2 Model Selection for Inference

The model selected for inference is determined **after** training by the following data-driven procedure:

1. Primary criterion: highest F1-score on the held-out test set
2. If two models are within 1% F1 of each other: prefer the more interpretable one (Random Forest over XGBoost) — consistent with Occam's Razor and the regularization principle from Lecture 4
3. Document the selection with reference to F1, ROC-AUC, and the business cost of false negatives

> Do not pre-commit to XGBoost. Let the experimental results determine selection.

### 8.3 Inference Interpretation

After displaying the inference results table, add a markdown cell that:
- Comments on whether the selected model's predictions align with the true labels
- Notes any misclassifications and hypothesizes why (e.g., borderline probability near 0.5)
- Connects back to the business use case: "If deployed, this model would have correctly flagged X/10 at-risk orders for proactive intervention"

---

## 9. Report (PDF Deliverable)

### 9.1 Format and Constraints

- **Length:** 7-12 pages (strict guideline requirement)
- **Format:** PDF only — no Word, no Google Docs export
- **Template:** Use the Overleaf template when shared on Moodle (strongly recommended per guidelines)
- **Style:** Research-project style — formal, third-person, structured sections with clear headings
- **Citation:** Cite the dataset (Mendeley DOI), scikit-learn, XGBoost, and any referenced lecture material or papers

### 9.2 Required Sections (Detailed Content Guide)

#### Introduction (~1-1.5 pages)
- **Business problem:** Late deliveries in e-commerce supply chains — cost to business, customer churn, operational inefficiency
- **Motivation:** Why ML-based early warning is valuable vs. rule-based thresholds
- **Project objective:** One clear sentence defining what the project predicts, using what data, for what purpose
- **Scope statement:** What this project covers and what it explicitly does not cover (e.g., no deep learning, no time-series forecasting, no deployment)

#### Background (~1.5-2 pages)
- **Dataset description:** DataCo dataset origin, structure, dimensions (marketing/logistics/financial), size, license
- **Domain context:** Brief overview of ML applications in supply chain management — mention 2-3 prior approaches or studies (not a full literature review, but enough to situate the work)
- **Feature space discussion:** Which features are available at order time vs. post-shipment, and why this distinction matters for the prediction task
- **Class distribution analysis:** Report the actual class balance with a bar chart or table; explain its implications for metric selection

#### Methodology (~2-3 pages)
- **Preprocessing pipeline:** Summarize all steps from Section 4 — missing values, duplicates, outliers, leakage removal, encoding, scaling. Justify each major decision. Do NOT paste code — describe the logic
- **Model selection rationale:** For each of the five models, explain in 2-3 sentences why it was included. Emphasize architectural diversity (instance-based vs. linear vs. margin-based vs. bagging vs. boosting)
- **Hyperparameter tuning strategy:** Explain GridSearchCV vs. RandomizedSearchCV choice and why. Mention the scoring metric (F1) and cross-validation folds (5)
- **Evaluation strategy:** Define all five metrics, justify why accuracy alone is insufficient for this problem, explain the business cost asymmetry (false negatives > false positives)
- **Scalability decisions:** Document the k-NN/SVM subsampling decision and why it is scientifically valid

#### Results & Interpretation (~2-3 pages)
- **Comparison table:** Reproduce the 5-model metric comparison table
- **Key figures:** Include ROC overlay, best confusion matrices (at least 2-3), and feature importance chart. Every figure must have a caption and be referenced in the text
- **Model-by-model analysis:** Brief commentary on each model's strengths, weaknesses, and surprises (e.g., "SGDClassifier's low recall confirms the decision boundary is not linearly separable")
- **Final model selection:** State which model was selected, with quantitative justification. Reference the selection procedure from Section 8.2
- **Inference results:** Include the 10-row inference table and interpret the outcomes

#### Conclusion (~0.5-1 page)
- **Summary of findings:** Which model won and why, what the feature importances reveal about the business problem
- **Limitations:** Acknowledge constraints — subsampling for k-NN/SVM, limited feature engineering, no temporal validation, no deployment testing
- **Future work:** 2-3 concrete suggestions (e.g., time-series features, cost-sensitive learning, model deployment as an API)

#### References
- DataCo dataset (Mendeley DOI)
- scikit-learn documentation
- XGBoost library
- Course lecture slides (Lecture 4)
- Any additional sources consulted

### 9.3 Report Quality Checklist

- [ ] All figures have captions and are referenced in text
- [ ] No raw code in the report (describe logic, not syntax)
- [ ] Consistent formatting throughout (headings, fonts, spacing)
- [ ] Page count is within 7-12 pages
- [ ] PDF renders correctly (no broken LaTeX, no missing images)
- [ ] All team member names appear on the title page
- [ ] Dataset citation is present

---

## 10. Notebook Structure (Presentation Artifact)

### 10.1 Why This Matters

Per guidelines: **"You will present your work primarily through your notebook, not through separate presentation slides."** The notebook IS the presentation. It must be fully executed before the presentation, with all outputs visible.

### 10.2 Required Notebook Structure

The notebook must follow this exact section flow. Each section begins with a bold Markdown heading cell.

| Section # | Heading                            | Content                                                                                     |
| :-------- | :--------------------------------- | :------------------------------------------------------------------------------------------ |
| 1         | **Project Overview**               | Problem statement, business objective, dataset summary, team members                        |
| 2         | **Data Loading & Initial Audit**   | Load CSV, display shape, dtypes, head(), describe(), missing values heatmap, class balance   |
| 3         | **Inference Holdout**              | Extract 10 rows, explain why this is done first                                              |
| 4         | **Data Quality & Preprocessing**   | Missing values, duplicates, outliers, leakage removal, identifier removal, type validation  |
| 5         | **Feature Encoding & Scaling**     | One-hot encoding, RobustScaler, document column count before/after                          |
| 6         | **Train-Test Split**               | Stratified 80/20, verify class balance in both sets                                          |
| 7         | **Model 1: k-NN**                  | Subsample justification, GridSearchCV, best params, test metrics, confusion matrix          |
| 8         | **Model 2: SGDClassifier**         | Training, GridSearchCV, best params, test metrics, confusion matrix                          |
| 9         | **Model 3: LinearSVC**             | CalibratedClassifierCV wrapper, GridSearchCV, best params, test metrics, confusion matrix   |
| 10        | **Model 4: Random Forest**         | RandomizedSearchCV, best params, test metrics, confusion matrix, feature importances         |
| 11        | **Model 5: XGBoost**               | RandomizedSearchCV, best params, test metrics, confusion matrix, feature importances         |
| 12        | **Model Comparison & Selection**   | Comparison table, ROC overlay, PR curve, selection justification                             |
| 13        | **Inference on Unseen Data**       | Apply pipeline to holdout, predict + predict_proba, display results table, interpret         |
| 14        | **Conclusion**                     | Summary of findings, limitations, what was learned                                           |

### 10.3 Notebook Quality Rules

- **Every code cell** must be preceded by a Markdown cell explaining what the code does and why
- **Every output cell** (table, plot, metric) must be followed by a Markdown cell interpreting the result
- **No orphan outputs** — a confusion matrix without interpretation is a missed grading opportunity
- **No commented-out code blocks** — clean up before submission
- **No error outputs** — the notebook must run cleanly from top to bottom
- **All cells must be executed** — the notebook must be submitted with all outputs visible (do not clear outputs)
- **Use `print()` statements** to display key metrics inline so they are visible without scrolling through large outputs

---

## 11. Presentation Preparation

### 11.1 Format (Per Guidelines)

- **Medium:** Present through the notebook — NOT through slides
- **Notebook state:** Must be fully executed before the presentation (all outputs visible)
- **Participants:** Preferable that more than one team member presents (split sections)
- **Duration:** To be communicated (prepare for ~15-20 minutes + Q&A as a safe assumption)
- **No live demo required** — do not run code live; walk through pre-executed outputs

### 11.2 Presentation Flow

Suggested speaker assignments (adapt to your team of 4-5):

| Section                         | Duration (est.) | Speaker   |
| :------------------------------ | :-------------- | :-------- |
| Problem definition & dataset    | 3 min           | Member 1  |
| Preprocessing pipeline          | 3 min           | Member 2  |
| Models overview & justification | 4 min           | Member 3  |
| Results, comparison, selection  | 4 min           | Member 4  |
| Inference & conclusion          | 2 min           | Member 5 (or Member 1) |
| Q&A                             | 5-10 min        | All       |

### 11.3 Q&A Defense Preparation (CRITICAL)

The guidelines state: **"You will be assessed not only on what you implement, but also on what you genuinely understand."** Every team member must be able to defend every algorithm.

**For each of the five models, every team member must be able to answer:**

1. What is the main idea of this model? (1-2 sentences)
2. What is the mathematical intuition? (explain the core formula)
3. What do the hyperparameters mean? (e.g., "What does k do in k-NN?")
4. How is the model evaluated? (which metrics, why those metrics)
5. Why was this model included in the comparison? (what does it contribute scientifically)

**Likely examiner questions to prepare for:**

- "Why did you use F1 instead of accuracy?" → Because the dataset is imbalanced; accuracy would mask poor recall on the minority class
- "Why did you subsample for k-NN?" → O(n·d) inference complexity; 180K rows would be prohibitively slow; subsampling is documented and stratified to preserve class balance
- "What is the difference between bagging and boosting?" → Bagging trains independent trees on bootstrap samples (parallel, reduces variance); boosting trains sequential trees on residual errors (sequential, reduces bias)
- "Why SGDClassifier instead of Perceptron?" → Perceptron lacks `predict_proba()`; SGDClassifier with modified_huber implements the soft threshold from Lecture 4
- "What does C do in SVM?" → Controls the regularization-margin tradeoff; low C = wide margin with more misclassifications; high C = narrow margin with fewer misclassifications
- "What is data leakage and how did you prevent it?" → Using post-shipment features to predict pre-shipment outcomes; we removed all columns only available after the order ships
- "Why not use a neural network?" → Course constraint (max 3 layers), and the problem is well-suited to classical ML on tabular data; no evidence a NN would outperform XGBoost here
- "What does `CalibratedClassifierCV` do?" → Wraps a classifier that doesn't support `predict_proba()` and fits a calibration model (Platt scaling or isotonic regression) to produce probability estimates
- "Why RobustScaler instead of StandardScaler?" → RobustScaler uses median and IQR, making it resistant to outliers; StandardScaler uses mean and std, which are distorted by extreme values

### 11.4 Presentation Dry Run

- Conduct at least one full dry run before presentation day
- Time each section to ensure you stay within limits
- Practice transitions between speakers
- Ensure every team member can navigate the notebook

---

## 12. Submission Logistics

### 12.1 What to Submit

| Deliverable           | Format     | Upload to |
| :-------------------- | :--------- | :-------- |
| Python notebook       | `.ipynb`   | Moodle    |
| Report                | `.pdf`     | Moodle    |

**No other formats will be considered official submissions.** Do not submit .py files, .docx files, datasets, or zipped folders unless explicitly requested.

### 12.2 Submission Deadline

**At most 1 hour before your assigned presentation slot.**

- If your presentation is at 10:00 AM, submit by 9:00 AM
- Do NOT wait until the last minute — upload issues happen
- Verify both files uploaded correctly on Moodle after submission

### 12.3 Pre-Submission Checklist

- [ ] Notebook runs from top to bottom without errors (Kernel → Restart & Run All)
- [ ] All outputs are visible (cells are executed, not cleared)
- [ ] No hardcoded absolute file paths (use relative paths for data loading)
- [ ] Report is exported as PDF (not .docx or .tex)
- [ ] Report is within 7-12 pages
- [ ] Both files are named clearly (e.g., `Team_SC1_Notebook.ipynb`, `Team_SC1_Report.pdf`)
- [ ] All team member names are in the notebook header and report title page
- [ ] Dataset is NOT included in submission (too large; provide download instructions in notebook)

---

## 13. Timeline and Task Assignment

### 13.1 Critical Dates

| Date       | Event                                       |
| :--------- | :------------------------------------------ |
| 2026-04-21 | Today — finalize plan, begin execution       |
| 2026-04-23 | Notebook code complete (all 5 models done)   |
| 2026-04-24 | Report draft complete                        |
| 2026-04-25 | Notebook polish, report finalize, dry run    |
| 2026-04-26 | Presentation week begins — submit 1h before slot |

### 13.2 Suggested Task Distribution (4-5 Members)

| Task                                      | Assignee(s)     | Deadline   |
| :---------------------------------------- | :-------------- | :--------- |
| Data loading, audit, preprocessing        | Member 1 + 2    | Apr 22     |
| k-NN + SGDClassifier (models 1 & 2)       | Member 1        | Apr 23     |
| LinearSVC + Random Forest (models 3 & 4)  | Member 2        | Apr 23     |
| XGBoost (model 5) + comparison table/plots| Member 3        | Apr 23     |
| Inference stage + conclusion               | Member 4        | Apr 23     |
| Report writing (Introduction + Background)| Member 3 + 5    | Apr 24     |
| Report writing (Methodology)              | Member 1 + 2    | Apr 24     |
| Report writing (Results + Conclusion)     | Member 4 + 5    | Apr 24     |
| Notebook annotation & polish              | All             | Apr 25     |
| Report review & PDF export                | All             | Apr 25     |
| Presentation dry run                      | All             | Apr 25     |
| Q&A defense preparation (study models)    | All (individual)| Apr 25     |

---

## 14. Compliance Checklist (Complete)

### 14.1 Technical Requirements

| Requirement                                   | Status     | Notes                                                                 |
| :-------------------------------------------- | :--------- | :-------------------------------------------------------------------- |
| At least 3 models with distinct architectures | Done       | 5 models across 4 families                                            |
| No deep learning / large neural networks      | Done       | No NN used; SGDClassifier is a single-layer linear model              |
| Python notebook (.ipynb), clean and annotated  | Planned    | Markdown above every step, interpretation below every output          |
| Hyperparameter tuning for all models           | Done       | GridSearchCV (3 models) + RandomizedSearchCV (2 models)               |
| Comparison table and plots mandatory           | Done       | ROC overlay + 5 confusion matrices + PR curve + feature importances   |
| Inference on unseen data                       | Done       | 10-row holdout set, held out before any preprocessing                 |
| Overfitting mitigation                         | Done       | 5-Fold Stratified CV                                                  |
| Lecture-grounded justification for models      | Done       | k-NN, SGDClassifier, SVM direct; RF/XGBoost via regularization/loss   |
| predict_proba() available for all models       | Done       | SGDClassifier(loss='modified_huber') + CalibratedClassifierCV for SVM |
| No data leakage                                | Done       | Leakage columns dropped; inference holdout before all preprocessing   |
| Scalability risks mitigated                    | Done       | 25K subsample for k-NN and SVM; RandomizedSearchCV for RF/XGBoost    |
| No AutoML, no pre-trained models               | Done       | All models manually configured                                        |
| No copy-paste / unmodified tutorial pipelines  | Planned    | Original implementation required                                      |

### 14.2 Report Requirements

| Requirement                                    | Status     | Notes                                          |
| :--------------------------------------------- | :--------- | :--------------------------------------------- |
| PDF format only                                | Planned    | Export from Overleaf or LaTeX                   |
| 7-12 pages                                     | Planned    | See Section 9 for detailed content guide        |
| Research-project style                         | Planned    | Use Overleaf template when available            |
| Introduction section                           | Planned    | Business problem, motivation, scope             |
| Background section                             | Planned    | Dataset, domain context, class distribution     |
| Methodology section                            | Planned    | Preprocessing, models, evaluation, scalability  |
| Results & Interpretation section               | Planned    | Comparison table, figures, model selection       |
| Conclusion section                             | Planned    | Findings, limitations, future work              |
| References section                             | Planned    | Dataset DOI, libraries, lecture material         |
| No raw code in report                          | Planned    | Describe logic, not syntax                      |
| All figures captioned and referenced           | Planned    | Every figure has a caption and in-text reference |

### 14.3 Presentation Requirements

| Requirement                                    | Status     | Notes                                           |
| :--------------------------------------------- | :--------- | :---------------------------------------------- |
| Present through notebook (not slides)          | Planned    | Notebook must be fully executed before presenting|
| All outputs visible in notebook                | Planned    | Kernel → Restart & Run All before submission     |
| Covers full workflow (problem → inference)      | Planned    | 14 notebook sections defined in Section 10       |
| Multiple team members present                  | Planned    | Speaker assignments in Section 11.2              |
| All members can defend all algorithms          | Planned    | Q&A prep guide in Section 11.3                   |
| No live demo required                          | Noted      | Walk through pre-executed outputs only           |

### 14.4 Submission Requirements

| Requirement                                    | Status     | Notes                                           |
| :--------------------------------------------- | :--------- | :---------------------------------------------- |
| Submit .ipynb + .pdf on Moodle                 | Planned    | No other formats accepted                        |
| Submit at most 1 hour before presentation slot | Planned    | Verify upload succeeded after submitting         |

---

## 15. Key Implementation Snippets

### Inference Holdout (top of notebook)

```python
df_inference = df.sample(10, random_state=42)
df = df.drop(df_inference.index).reset_index(drop=True)
```

### SVM with predict_proba support

```python
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
svm_base = LinearSVC(max_iter=5000, random_state=42)
svm_calibrated = CalibratedClassifierCV(svm_base, cv=5)
```

### RandomizedSearchCV for ensemble models

```python
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(
    estimator, param_distributions, n_iter=20,
    cv=5, scoring='f1', random_state=42, n_jobs=-1
)
```

### k-NN subsample

```python
from sklearn.utils import resample
X_train_knn, y_train_knn = resample(
    X_train, y_train, n_samples=25000,
    stratify=y_train, random_state=42
)
```

---

## 16. Academic Integrity Note

AI tools may be used as support tools only. All code, model choices, and interpretations must be understood and defensible during the presentation. The notebook must demonstrate original authorship. Every algorithm must be explainable: core idea, mathematical intuition, hyperparameter meanings, and reason for inclusion in this comparison.

**Prohibited:** copy-paste submissions, copied notebooks, unmodified tutorial pipelines, pre-trained models, AutoML-based solutions, code that cannot be explained and defended.

---

_Document prepared for CS280/CS485 Introduction to Artificial Intelligence — Mediterranean Institute of Technology — Spring 2026_
