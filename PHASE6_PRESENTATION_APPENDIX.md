# Phase 6 — Presentation Style Appendix

**Applies to:** `notebooks/07_final_merged.ipynb` only  
**Constraint:** Lab guidelines require one fully executable notebook submitted as the sole deliverable. This appendix ensures the notebook satisfies that requirement while remaining clean to scroll through during a 15-minute presentation.

---

## 1. Core rules

1. `Run All` must work top-to-bottom with no errors in a clean environment where only `artifacts/prepared_data.pkl` exists. No other file may be a prerequisite.
2. Cell collapse is purely cosmetic — collapsed cells still execute. Never reorder cells in a way that breaks execution order to serve the presentation flow.
3. All model training happens inline in this notebook. Do not load pre-trained models from pkl files.

---

## 2. Cell visibility defaults

Apply these defaults to every cell before saving:

| Cell type | Default state |
|-----------|--------------|
| Import / setup code | **Collapsed** |
| Data loading code | **Collapsed** |
| Hyperparameter grid definitions | **Collapsed** |
| GridSearch / RandomizedSearch / fit code | **Collapsed** |
| Verbose GridSearch output (cv_results_, best_score_ logs) | **Collapsed** |
| Metrics computation code | **Collapsed** |
| `classification_report` text output | **Collapsed** |
| Per-model "Lecture 4 connection" markdown sections (detailed) | **Collapsed** (open on demand during Q&A) |
| Checkpoint save-to-pkl code | **Collapsed** |
| Section header markdown cells | **Open** |
| One-line summary markdown (best params + F1 + ROC-AUC) | **Open** |
| Confusion matrix plot outputs | **Open** |
| Feature importance chart outputs (RF and XGBoost only) | **Open** |
| k vs. F1 validation curve (k-NN only) | **Open** |
| All comparison section outputs (table, ROC, PR, bar chart, CV box plot) | **Open** |
| Inference results table | **Open** |
| Narrative markdown between outputs | **Open** |
| Conclusion markdown | **Open** |

In JupyterLab, collapsing is done by clicking the blue vertical bar to the left of the cell. The collapsed state is saved in the notebook JSON and persists across sessions.

---

## 3. Section order (presentation-first, execution-safe)

The notebook tells a story when scrolled top-to-bottom. Execution order and presentation order are identical — no reordering needed.

1. **Title + team + problem statement** — markdown only
2. **Imports** — collapsed
3. **Load `artifacts/prepared_data.pkl`** — collapsed code, one-line confirmation output open
4. **Preprocessing summary** — open markdown: key decisions only (leakage removal, RobustScaler, stratified split). No preprocessing code runs here — it ran in Phase 0 and its output is already in the pkl.
5. **k-NN** — train inline, outputs open per §4 below
6. **SGDClassifier** — train inline, outputs open per §4 below
7. **LinearSVC** — train inline, outputs open per §4 below
8. **Random Forest** — train inline, outputs open per §4 below
9. **XGBoost** — train inline, outputs open per §4 below
10. **Model comparison** — built from in-memory variables, all outputs open per §5 below
11. **Inference** — 10-row holdout table, open
12. **Conclusion** — winner declaration, open
13. **Appendix** — verbose logs, raw metric printouts, any extra diagnostics, all collapsed

---

## 4. Per-model section template

Each model section (steps 5-9 above) must follow this exact cell sequence:

```
[OPEN]      ## Model Name — one-sentence algorithm description
[COLLAPSED] Lecture 4 connection — detailed markdown (formulas, mapping to lecture symbols)
[COLLAPSED] Hyperparameter grid definition
[COLLAPSED] GridSearch / RandomizedSearch / fit code
[COLLAPSED] Verbose GridSearch output
[OPEN]      One-line summary: Best params | Train F1 | Test F1 | ROC-AUC
[COLLAPSED] Metrics computation code (accuracy, precision, recall, f1, roc_auc)
[OPEN]      Confusion matrix plot
[OPEN]      Model-specific plot (k vs. F1 curve for k-NN; feature importance for RF and XGBoost)
[COLLAPSED] 5-fold CV scores printout
[COLLAPSED] Checkpoint save to results/<tag>_results.pkl
```

Each model section takes ~1 minute to present: read the header, state the one-line summary, point to the confusion matrix. Move on.

---

## 5. Comparison section template

Built from in-memory variables — no pkl reloading.

```
[OPEN]      ## Model Comparison
[COLLAPSED] Code: build comparison DataFrame from in-memory metric variables
[OPEN]      5-model metric table output
[OPEN]      ROC-curve overlay output (AUCs in legend)
[OPEN]      Precision-Recall curve overlay output
[OPEN]      Train vs. Test F1 bar chart output (5 models × 2 bars)
[OPEN]      5-fold CV F1 box plot output
[OPEN]      Narrative markdown: winner, reason, Lecture 4 overfitting framing
```

This is the climax of the presentation. Spend 2 minutes here.

---

## 6. What NOT to show during presentation (but must remain in notebook)

These must exist for the grader but must be collapsed before walking into the room:

- All `import` blocks
- Raw `GridSearchCV` / `RandomizedSearchCV` fit calls and their verbose output
- `best_params_` raw printouts (summarized in the one-line open summary cell instead)
- `classification_report` text output (confusion matrix plot replaces it visually)
- 5-fold CV score printouts (visible in the CV box plot instead)
- Checkpoint save-to-pkl cells
- Any cell with more than ~10 lines of output
- The detailed "Lecture 4 connection" markdown sections (closed, recite from memory if asked)

---

## 7. Pre-presentation checklist

Run through this the night before:

- [ ] Delete all `results/*.pkl` files and `results/comparison_table.csv`, then hit `Run All` — the notebook must complete with no errors from a clean state
- [ ] Restore the pkl files after the clean-run test (or re-run all phases)
- [ ] Scroll top-to-bottom — only intended cells are open, no walls of text visible
- [ ] Every model section shows: header + one-line summary + confusion matrix. Nothing else.
- [ ] Comparison section shows all five charts and the winner declaration
- [ ] Inference table shows true label, predicted label, and probability for all 10 rows
- [ ] "Lecture 4 connection" sections are collapsed but every team member can recite each one
- [ ] Notebook is saved with all outputs — do not clear outputs before presenting
