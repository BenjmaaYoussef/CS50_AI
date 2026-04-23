# PHASES.md — Parallel Execution Plan

**Project:** DataCo Smart Supply Chain — Late Delivery Risk Classification
**Course:** CS280 / CS485 — Introduction to AI, Spring 2026
**Parent plan:** `docs.md` (the technical plan — all decisions there are authoritative)
**Governing documents:** `CS280-CS485 Lab Project Guidelines S26.pdf`, `lecture4.pdf`

---

## 0. Verification Result of `docs.md` (before splitting into phases)

`docs.md` (786 lines) was audited end-to-end against the two PDFs. Findings:

| Area | Status | Note |
| :--- | :--- | :--- |
| sklearn / XGBoost API calls | ✅ Correct | `SGDClassifier(loss='modified_huber')`, `CalibratedClassifierCV(..., estimator__C)`, `resample(replace=False)`, `learning_rate='constant'` → all verified against real APIs. |
| Math (k-NN distance, perceptron rule, SVM margin, cost(h)=loss(h)+λ·complexity(h), XGBoost objective) | ✅ Correct | No hallucinated formulas. |
| Lecture 4 concept references (perceptron hard/soft threshold, SVM max-margin, loss functions, overfitting, regularization framework) | ✅ Match | Confirmed against `lecture4.pdf`. |
| Lab-guideline requirements (≥3 distinct models, no deep learning, notebook + 7-12 pg PDF, inference on unseen data, CV, comparison tables/plots, academic integrity) | ✅ Covered | |
| Scalability fixes (k-NN 25K subsample, LinearSVC for SVM) | ✅ Valid | Justified and documented. |

### Algorithm-selection rule (professor's constraint, clarified)

The professor's rule is: **the models studied in class are the mandatory backbone; up to 2 additional algorithms may be outside class as long as they follow the lab guidelines.**

`docs.md`'s 5-model lineup matches this rule exactly:

| # | Model | In Lecture 4? | Role |
| :- | :--- | :--- | :--- |
| 1 | k-NN | ✅ Directly (nearest-neighbor classification) | In-class core |
| 2 | SGDClassifier (Perceptron / soft threshold) | ✅ Directly (perceptron + soft threshold + learning rule) | In-class core |
| 3 | LinearSVC (maximum-margin separator) | ✅ Directly (max-margin classifier + L2 regularization) | In-class core |
| 4 | Random Forest | ❌ Not named | 1st allowed extension — justified via Lecture 4's regularization framework |
| 5 | XGBoost | ❌ Not named | 2nd allowed extension — justified via Lecture 4's regularization framework |

**Conclusion: plan is compliant — 3 in-class + 2 extensions. Proceed with all five phases as written.**

Defense posture for the two extensions: every Phase 4/5 notebook must contain the "Lecture 4 connection" section (§2.4 below) that ties the algorithm's hyperparameters to the in-class regularization formula `cost(h) = loss(h) + λ·complexity(h)` — i.e., RF's `max_depth`/`min_samples_leaf` and XGBoost's `γT + ½λ‖w‖²` are concrete instantiations of that framework. This is how the team defends *why* a non-lecture algorithm was chosen.

---

## 1. Phase Graph

```
              ┌──────────────────────────────────┐
              │  Phase 0  —  Preprocessing       │   (SEQUENTIAL · BLOCKER)
              │  produces prepared_data.pkl      │
              └──────────────┬───────────────────┘
                             │
     ┌──────────┬────────────┼────────────┬──────────┐
     ▼          ▼            ▼            ▼          ▼
 ┌───────┐  ┌────────┐  ┌──────────┐ ┌────────┐ ┌─────────┐
 │ Ph. 1 │  │ Ph. 2  │  │  Ph. 3   │ │ Ph. 4  │ │  Ph. 5  │
 │ k-NN  │  │ SGDClf │  │ LinearSVC│ │  RF    │ │ XGBoost │    (PARALLEL · NO CONFLICTS)
 └───┬───┘  └───┬────┘  └────┬─────┘ └───┬────┘ └────┬────┘
     │          │            │           │           │
     └──────────┴────────────┼───────────┴───────────┘
                             ▼
              ┌──────────────────────────────────┐
              │  Phase 6  —  Merge & Deliver     │   (USER RUNS MANUALLY
              │  comparison + inference + PDF    │    after collecting all
              │                                  │    model results)
              └──────────────────────────────────┘
```

Phases 1-5 touch **disjoint files** — they can run simultaneously in five separate Claude Code sessions without merge conflicts.

---

## 2. Shared conventions (read before starting any phase)

### 2.1 Directory layout

```
CS50_AI/
├── docs.md                          # authoritative plan — every phase cites it
├── PHASES.md                        # this file
├── lecture4.pdf
├── CS280-CS485 Lab Project Guidelines S26.pdf
├── data/
│   └── DataCoSupplyChainDataset.csv           # raw dataset (team drops here first)
├── artifacts/
│   ├── prepared_data.pkl                      # produced by Phase 0
│   └── pipeline_metadata.json                 # column names, scaler/encoder refs
├── notebooks/
│   ├── 01_preprocessing.ipynb                 # Phase 0
│   ├── 02_model_knn.ipynb                     # Phase 1
│   ├── 03_model_sgd.ipynb                     # Phase 2
│   ├── 04_model_svm.ipynb                     # Phase 3
│   ├── 05_model_rf.ipynb                      # Phase 4
│   ├── 06_model_xgb.ipynb                     # Phase 5
│   └── 07_final_merged.ipynb                  # Phase 6 (user-produced)
├── results/
│   ├── knn_results.pkl                        # metrics, best_params, fitted model
│   ├── sgd_results.pkl
│   ├── svm_results.pkl
│   ├── rf_results.pkl
│   └── xgb_results.pkl
└── report/
    └── report.pdf                             # Phase 6
```

No two parallel phases write to the same file. Each Phase 1-5 owns exactly one notebook and one `*_results.pkl`.

### 2.2 Universal reproducibility rules (every phase)

- `random_state=42` everywhere that accepts it.
- Load exclusively from `artifacts/prepared_data.pkl` — **never reload or re-split the raw CSV** in Phase 1-5.
- Fit no preprocessing objects (scaler, encoder) in Phase 1-5 — they are already fit in Phase 0.
- Every code cell preceded by a Markdown cell explaining *what* and *why*.
- Every output (table/plot/metric) followed by a Markdown interpretation cell.
- No `!pip install` in notebooks committed to the repo — list packages in `requirements.txt`.
- Notebook must run top-to-bottom with no errors, all outputs saved.

### 2.3 `*_results.pkl` schema (mandatory — Phase 6 depends on this)

Each Phase 1-5 saves a dict with these **exact keys**:

```
{
    "model_name":      str,          # "k-NN" | "SGDClassifier" | "LinearSVC" | "RandomForest" | "XGBoost"
    "model":           fitted_estimator,   # the best estimator from GridSearch/RandomizedSearch
    "best_params":     dict,
    "y_pred":          np.ndarray,   # predictions on X_test
    "y_proba":         np.ndarray,   # predict_proba(X_test)[:, 1]
    "metrics": {
        "accuracy": float, "precision": float, "recall": float,
        "f1": float, "roc_auc": float
    },
    "cv_f1_scores":    np.ndarray,   # 5-fold stratified CV F1 on training set
    "train_f1":        float,        # F1 on X_train (for overfitting chart)
    "test_f1":         float,
    "feature_importance": dict | None   # {feature_name: score} if available, else None
}
```

Any deviation breaks Phase 6. Confirm keys before pickling.

### 2.4 Lecture-4 citation discipline

Every model notebook **must contain** a Markdown section titled "Lecture 4 connection" that:
- Names the concept(s) from Lecture 4 relevant to the algorithm (e.g., nearest-neighbor classification, perceptron learning rule, maximum-margin separator, regularization framework).
- Writes the relevant equation.
- States in one sentence how the sklearn/XGBoost hyperparameters map onto the lecture's symbols (especially λ ↔ `alpha` / `C` / `reg_lambda`).

This is graded defensively — the professor can ask any team member to justify *any* algorithm.

---

## 3. Phase 0 — Preprocessing (BLOCKER · runs first · one session)

**Owner:** 1 person · **Inputs:** `data/DataCoSupplyChainDataset.csv` · **Outputs:** `notebooks/01_preprocessing.ipynb`, `artifacts/prepared_data.pkl`, `artifacts/pipeline_metadata.json`

### Prompt to feed Claude Code

> Implement Sections 2, 3, and 4 of `docs.md` as the notebook `notebooks/01_preprocessing.ipynb`. Follow Section 10.2 rows 1-6 verbatim for the notebook structure. The notebook must produce and pickle an `artifacts/prepared_data.pkl` containing `{X_train, X_test, y_train, y_test, df_inference_raw, df_inference_processed, scaler, encoder, feature_names, target_name}`. Respect Section 4.0 (10-row inference holdout BEFORE any preprocessing), Section 4.2 (data-leakage column removal — any column observable only post-shipment), Section 4.3 (programmatic identifier removal, not hard-coded lists), Section 4.5 (RobustScaler fit on train only), and Section 4.6 (stratified 80/20 split, random_state=42). Include the four preprocessing charts from Section 7.3.A (class distribution, missing-values bar chart, correlation heatmap, outlier box plots). Every decision gets a markdown justification above, every output gets an interpretation below. Do not start any modeling in this notebook — it ends at the end of Section 6 of the notebook structure.

### Success criteria
- [ ] `df_inference` held out **before** any fit.
- [ ] `leakage_columns_dropped` logged with a one-line justification per column.
- [ ] Scaler is fit on `X_train` only, applied to `X_test` and `df_inference_processed`.
- [ ] Target column is `Late_delivery_risk` (verify not dropped or leaked).
- [ ] `prepared_data.pkl` loads cleanly in a fresh Python session.
- [ ] Notebook runs top-to-bottom with no errors and all four charts render.

### Guardrails
- If the team realizes a preprocessing decision needs changing **after** Phase 0 ships, re-run Phase 0, then **re-run all Phase 1-5 notebooks**. Do not edit preprocessing inside model phases.

---

## 4. Phases 1-5 — Model phases (PARALLEL · five separate sessions)

Each phase uses the same template. Fill in the model-specific row from the table below.

### 4.1 Shared template (identical structure for all five)

1. Load `artifacts/prepared_data.pkl`.
2. Build the model with the scalability / API fix from `docs.md`.
3. Run the hyperparameter search listed.
4. Report best_params, train F1, test metrics (accuracy, precision, recall, F1, ROC-AUC), confusion matrix.
5. Run 5-fold Stratified CV on training set (`scoring='f1'`).
6. Produce model-specific plots (see table).
7. Write "Lecture 4 connection" markdown section per §2.4.
8. Save to `results/<tag>_results.pkl` using the schema in §2.3.

### 4.2 Phase-by-phase dispatch table

| # | Model | Notebook | Results file | docs.md section | Required API fix | Search | Must-produce plots (beyond confusion matrix) |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | k-NN | `notebooks/02_model_knn.ipynb` | `results/knn_results.pkl` | 5.1 | `resample(..., replace=False)` to 25K rows, `algorithm='ball_tree'`, `n_jobs=-1` | GridSearchCV (k, metric, weights) | **k vs. validation F1 curve** (Fig 8 of 7.3.B) |
| 2 | SGDClassifier (Perceptron) | `notebooks/03_model_sgd.ipynb` | `results/sgd_results.pkl` | 5.2 | `SGDClassifier(loss='modified_huber', learning_rate='constant', ...)` | GridSearchCV (alpha, eta0, max_iter) | none beyond shared set |
| 3 | LinearSVC (SVM) | `notebooks/04_model_svm.ipynb` | `results/svm_results.pkl` | 5.3 | `CalibratedClassifierCV(LinearSVC(...))` + grid uses `estimator__C`, `estimator__max_iter` | GridSearchCV | none beyond shared set |
| 4 | Random Forest | `notebooks/05_model_rf.ipynb` | `results/rf_results.pkl` | 5.4 | standard `RandomForestClassifier` (no scaling needed but pipeline applies it) | **RandomizedSearchCV n_iter=20** | **n_estimators vs. F1 curve**, **feature-importance bar chart (top 15)** |
| 5 | XGBoost | `notebooks/06_model_xgb.ipynb` | `results/xgb_results.pkl` | 5.5 | `XGBClassifier(..., eval_metric='logloss', n_jobs=-1)`, handle class imbalance via `scale_pos_weight` if imbalanced | **RandomizedSearchCV n_iter=20** | **feature-importance bar chart (top 15)** |

### 4.3 Per-phase prompts (copy-paste into each parallel Claude Code session)

#### Phase 1 prompt — k-NN
> Implement Section 5.1 of `docs.md` as `notebooks/02_model_knn.ipynb`. Load `artifacts/prepared_data.pkl` as the sole data source. Apply the 25K `resample(..., replace=False, stratify=y_train, random_state=42)` subsample (bug-prevention note in docs.md). Use `KNeighborsClassifier(algorithm='ball_tree', n_jobs=-1)` inside `GridSearchCV(cv=5, scoring='f1', n_jobs=-1)` over the grid in the docs.md table. Produce the confusion matrix, the **k vs. F1 validation curve**, and 5-fold Stratified CV F1 scores on the *subsampled* training set. Include a "Lecture 4 connection" section naming nearest-neighbor classification as the direct lecture concept and writing the Euclidean distance + majority-vote formulas. Save to `results/knn_results.pkl` using the schema in PHASES.md §2.3. Do not touch any file outside `notebooks/02_model_knn.ipynb` and `results/knn_results.pkl`.

#### Phase 2 prompt — SGDClassifier
> Implement Section 5.2 of `docs.md` as `notebooks/03_model_sgd.ipynb`. Load `artifacts/prepared_data.pkl`. Use `SGDClassifier(loss='modified_huber', learning_rate='constant', random_state=42)` inside `GridSearchCV(cv=5, scoring='f1', n_jobs=-1)` over the docs.md grid (`alpha`, `eta0`, `max_iter`). Produce confusion matrix, train F1 and test metrics, 5-fold CV F1. Include a "Lecture 4 connection" section covering the perceptron concept from Lecture 4 — write the hard-threshold formula, the soft-threshold motivation, and the perceptron learning rule `w_i ← w_i + α(y − h_w(x))·x_i`, and map the `alpha` hyperparameter to λ in the regularization formula cost(h) = loss(h) + λ·complexity(h). Save to `results/sgd_results.pkl` using the schema in PHASES.md §2.3. Do not touch any file outside `notebooks/03_model_sgd.ipynb` and `results/sgd_results.pkl`.

#### Phase 3 prompt — LinearSVC
> Implement Section 5.3 of `docs.md` as `notebooks/04_model_svm.ipynb`. Load `artifacts/prepared_data.pkl`. Wrap `LinearSVC(random_state=42)` with `CalibratedClassifierCV(estimator=LinearSVC(...), cv=5)` so `predict_proba` works (required for ROC/PR). **Critical:** the GridSearchCV param grid must use `estimator__C` and `estimator__max_iter` — not `C`/`max_iter` — otherwise sklearn raises "Invalid parameter C". Use `GridSearchCV(cv=5, scoring='f1', n_jobs=-1)`. Produce confusion matrix, metrics, 5-fold CV F1. Include a "Lecture 4 connection" section naming the maximum-margin separator concept — write `min ½‖w‖² s.t. yᵢ(w·xᵢ+b) ≥ 1` and explain how `C` is the inverse of λ in the lecture's regularization formula cost(h) = loss(h) + λ·complexity(h). Save to `results/svm_results.pkl` using the schema in PHASES.md §2.3. Do not touch any file outside `notebooks/04_model_svm.ipynb` and `results/svm_results.pkl`.

#### Phase 4 prompt — Random Forest
> Implement Section 5.4 of `docs.md` as `notebooks/05_model_rf.ipynb`. Load `artifacts/prepared_data.pkl`. Use `RandomForestClassifier(random_state=42, n_jobs=-1)` inside `RandomizedSearchCV(n_iter=20, cv=5, scoring='f1', random_state=42, n_jobs=-1)` over the docs.md grid. Produce confusion matrix, metrics, 5-fold CV F1, **n_estimators vs. F1 curve**, and **feature-importance bar chart (top 15)** using `best_estimator_.feature_importances_` mapped to `feature_names` from the pickle. Include a "Lecture 4 connection" section explicitly flagging RF as one of the two allowed out-of-class extensions: tie `max_depth` / `min_samples_leaf` to the complexity term in the lecture's cost(h) = loss(h) + λ·complexity(h), and tie bagging's variance reduction to Lecture 4's overfitting discussion. Save to `results/rf_results.pkl` using the schema in PHASES.md §2.3. Do not touch any file outside `notebooks/05_model_rf.ipynb` and `results/rf_results.pkl`.

#### Phase 5 prompt — XGBoost
> Implement Section 5.5 of `docs.md` as `notebooks/06_model_xgb.ipynb`. Load `artifacts/prepared_data.pkl`. Use `XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)` — compute `scale_pos_weight = sum(y_train==0)/sum(y_train==1)` and pass it if class balance requires it. Run `RandomizedSearchCV(n_iter=20, cv=5, scoring='f1', random_state=42, n_jobs=-1)` over the docs.md grid (`n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`). Produce confusion matrix, metrics, 5-fold CV F1, and **feature-importance bar chart (top 15)**. Include a "Lecture 4 connection" section explicitly flagging XGBoost as one of the two allowed out-of-class extensions: write the XGBoost objective `ℒ(t) = Σ ℓ(yᵢ, ŷᵢ(t-1) + fₜ(xᵢ)) + Ω(fₜ)` with `Ω(fₜ) = γT + ½λ‖w‖²` and identify it as a literal instance of Lecture 4's cost(h) = loss(h) + λ·complexity(h). Save to `results/xgb_results.pkl` using the schema in PHASES.md §2.3. Do not touch any file outside `notebooks/06_model_xgb.ipynb` and `results/xgb_results.pkl`.

### 4.4 Per-phase checklist (each session must tick all)

- [ ] Loads only `artifacts/prepared_data.pkl` — no raw-CSV access.
- [ ] No scaler / encoder fitting.
- [ ] `random_state=42` on model, search, and any resampling.
- [ ] GridSearch / RandomizedSearch uses `scoring='f1'`, `cv=5`, `n_jobs=-1`.
- [ ] Confusion matrix printed and plotted.
- [ ] 5-fold Stratified CV scores recorded as `cv_f1_scores`.
- [ ] `train_f1` and `test_f1` both recorded (for the overfitting chart in Phase 6).
- [ ] "Lecture 4 connection" markdown present, naming the relevant lecture concept(s).
- [ ] `results/<tag>_results.pkl` follows §2.3 schema exactly.
- [ ] Notebook runs top-to-bottom with no errors.

---

## 5. Phase 6 — Merge, Inference & Deliverables (USER-OPERATED)

This phase is deliberately scoped to the user because it needs the five `results/*_results.pkl` files to exist. When the team hands those to Claude, the prompt below will produce the final artifacts.

### 5.1 Inputs expected

- `artifacts/prepared_data.pkl` — the only required external dependency, produced by Phase 0.

The phase notebooks (`02` through `06`) are **development references only** — they are not read, imported, or depended on by Phase 6. Their pkl outputs (`results/*.pkl`) are not prerequisites. Phase 6 trains all models inline from scratch.

### 5.2 Outputs

- `notebooks/07_final_merged.ipynb` — the single submitted notebook. Fully self-contained: trains all models inline, produces all charts, runs inference. A grader opening this file in a clean environment with only `artifacts/prepared_data.pkl` present must be able to hit "Run All" with no errors.
- `results/comparison_table.csv` — the 5-model metric summary (Section 7.5 of `docs.md`).
- `results/*_results.pkl` — optional mid-notebook checkpoints saved after each model trains. The notebook produces these; it does not depend on them existing beforehand.
- `README.md` — run instructions and team roster.

> **PDF report is out of scope for this phase.** It will be produced separately after the notebook is confirmed correct.

### 5.3 Phase-6 prompt (to run after inputs are assembled)

> Build `notebooks/07_final_merged.ipynb` as a single, fully self-contained notebook. The only external dependency is `artifacts/prepared_data.pkl` — do not load from any pre-existing `results/*.pkl` file and do not read or reference any phase notebook file. The notebook must run top-to-bottom with no errors in a clean environment where only `artifacts/prepared_data.pkl` exists.
>
> **Data loading:** Load `artifacts/prepared_data.pkl` once at the top. Extract `X_train`, `X_test`, `y_train`, `y_test`, `feature_names`, `scaler`, `encoder`, `df_inference_raw`, and `df_inference_processed`.
>
> **Train all five models inline in this order:**
> 1. **k-NN** — subsample to 25K rows with `resample(..., replace=False, stratify=y_train, random_state=42)`, use `KNeighborsClassifier(algorithm='ball_tree', n_jobs=-1)` inside `GridSearchCV(cv=5, scoring='f1', n_jobs=-1)` per Section 5.1. Produce the k vs. validation F1 curve.
> 2. **SGDClassifier** — `SGDClassifier(loss='modified_huber', learning_rate='constant', random_state=42)` inside `GridSearchCV(cv=5, scoring='f1', n_jobs=-1)` per Section 5.2.
> 3. **LinearSVC** — `CalibratedClassifierCV(LinearSVC(random_state=42), cv=5)` inside `GridSearchCV(cv=5, scoring='f1', n_jobs=-1)` with `estimator__C` and `estimator__max_iter` in the param grid per Section 5.3.
> 4. **Random Forest** — `RandomForestClassifier(random_state=42, n_jobs=-1)` inside `RandomizedSearchCV(n_iter=20, cv=5, scoring='f1', random_state=42, n_jobs=-1)` per Section 5.4. Produce the n_estimators vs. F1 curve and top-15 feature importance chart.
> 5. **XGBoost** — `XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)` with `scale_pos_weight` inside `RandomizedSearchCV(n_iter=20, cv=5, scoring='f1', random_state=42, n_jobs=-1)` per Section 5.5. Produce the top-15 feature importance chart.
>
> For each model: record `best_params`, `train_f1`, `test_f1`, all five metrics (accuracy, precision, recall, F1, ROC-AUC), and `cv_f1_scores` (5-fold stratified CV on training set). Produce the confusion matrix. Optionally checkpoint the model to `results/<tag>_results.pkl` immediately after training — but the notebook must not depend on these files existing at the start. Include a "Lecture 4 connection" markdown section per PHASES.md §2.4 for each model.
>
> **After all five models are trained**, build using in-memory variables (not by reloading pkl files): (a) **5-model comparison table** per Section 7.5, (b) **ROC-curve overlay** with AUCs in the legend, (c) **Precision-Recall curve overlay**, (d) **Train-vs-Test F1 bar chart** (5 models × 2 bars), (e) **5-fold CV box plot** across models. Save the comparison table as `results/comparison_table.csv`.
>
> **Inference Stage** (Section 8): apply the saved scaler/encoder from `prepared_data.pkl` to `df_inference_raw` — never refit. Run `.predict` and `.predict_proba` of the best-performing model selected by Section 8.2's data-driven rule. Display a results table with true label, predicted label, and risk probability.
>
> Cite Lecture 4 by concept throughout (nearest-neighbor, perceptron + learning rule, maximum-margin separator, loss functions, regularization) rather than by slide number.
>
> **Apply the presentation style conventions in `PHASE6_PRESENTATION_APPENDIX.md` in full** — collapsed cells, open key outputs, appendix section at the bottom — so the notebook is ready to present without any modification after "Run All".

### 5.4 Final compliance pass (team does last)

Re-open `docs.md` Section 14 (Compliance Checklist) and tick every item against the finished deliverables before submission. Specifically:

- [ ] ≥ 3 distinct models (we have 5).
- [ ] No deep learning / no pre-trained models (SGDClassifier is a single-layer linear classifier, confirmed).
- [ ] Notebook runs cleanly top-to-bottom, all outputs visible.
- [ ] PDF is 7-12 pages, PDF format only.
- [ ] Every algorithm has a "Lecture 4 connection" traceable to a named lecture concept.
- [ ] Inference on 10 held-out rows present and interpreted.
- [ ] Dataset cited (Mendeley DOI); scikit-learn and XGBoost cited.
- [ ] Every team member listed on title page.
- [ ] All team members can defend every algorithm (dry-run before presentation).

---

## 6. Role assignments (suggested, 4-5 members)

| Role | Phase(s) owned | Notes |
| :--- | :--- | :--- |
| **Data lead** | Phase 0 | Must finish first. After finishing, helps whoever lags on Phase 1-5. |
| **Model engineer A** | Phase 1 (k-NN) + Phase 2 (SGDClassifier) | Both are linear / instance-based — similar debugging surface. |
| **Model engineer B** | Phase 3 (LinearSVC) | The CalibratedClassifierCV wrapper is the fiddliest; assign the strongest sklearn user. |
| **Model engineer C** | Phase 4 (Random Forest) + Phase 5 (XGBoost) | Ensembles share hyperparameter structure. |
| **Report & presentation lead** | Phase 6 deliverables — writes report, prepares notebook narration, runs dry-run | Also owns the compliance checklist. |

If the team is 4 people, collapse Model engineers A+B or merge report lead with Data lead post-Phase-0.

---

## 7. Integration risks to watch

| Risk | Mitigation |
| :--- | :--- |
| Phase 0 is rerun after Phase 1-5 have already run → stale `prepared_data.pkl` → stale results. | Only rerun Phase 0 if a preprocessing bug is confirmed. After any rerun, **all 5 model phases must be rerun**. |
| Two phases both bump a shared pinned library version (e.g., `scikit-learn`). | Freeze versions in `requirements.txt` before Phase 0 starts. No model phase edits that file. |
| A phase silently deviates from the `*_results.pkl` schema. | Phase 6 prompt begins with a schema validator that loads each pickle and `assert`s the key set. |
| Q&A challenge: "why include RF/XGBoost if they weren't in Lecture 4?" | Every Phase 4/5 notebook must explicitly frame these as the "2 extensions" allowed by the professor's rule, and tie their regularization terms to the Lecture 4 cost(h) = loss(h) + λ·complexity(h) formula. Rehearse this answer in the dry-run. |

---

**End of PHASES.md.** When every model phase has shipped its `results/*_results.pkl`, hand the whole `CS50_AI/` tree back to Claude with the Phase 6 prompt and the merge will produce the final notebook + report.
