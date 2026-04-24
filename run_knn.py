"""
Phase 1 k-NN computation script.
Key design choices vs. naive implementation:
  - return_train_score=False  (default) — omitting it was causing 5x slowdown by
    predicting on the full 20K training fold in addition to the 5K validation fold.
  - n_jobs=1 in knn_base inside GridSearchCV — 16 worker processes already use all
    cores; adding 16 KNN threads per process creates 256 threads on 16 cores and
    degrades performance via thread contention.
  - n_jobs=-1 on the final fitted model — once GridSearchCV is done and we need
    three large prediction passes (36K test + 36K proba + 25K train), using all
    cores for KNN queries cuts evaluation from ~14 min to ~90 seconds.
"""
import pickle, time, warnings, sys, os
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.utils import resample

def log(msg):
    print(msg, flush=True)

log("=== k-NN Phase 1 Computation Script ===")
log(f"Start: {time.strftime('%H:%M:%S')}")

# 1. Load data
log("\n[1] Loading artifacts/prepared_data.pkl ...")
with open('artifacts/prepared_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train       = data['X_train']
X_test        = data['X_test']
y_train       = data['y_train']
y_test        = data['y_test']
feature_names = data['feature_names']

log(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
log(f"  y_train class balance -- 0:{(y_train==0).sum()}, 1:{(y_train==1).sum()}")
log(f"  y_test  class balance -- 0:{(y_test==0).sum()},  1:{(y_test==1).sum()}")
log(f"  Features: {len(feature_names)}")

# 2. Stratified 25K subsample (replace=False prevents duplicate rows)
log("\n[2] Stratified 25K subsample (replace=False) ...")
X_knn, y_knn = resample(X_train, y_train, n_samples=25000,
                          stratify=y_train, replace=False, random_state=42)
log(f"  Subsample shape: {X_knn.shape}")
log(f"  Class ratio -- full:{y_train.mean():.4f}  subsample:{y_knn.mean():.4f}")

# 3. GridSearchCV
# n_jobs=1 in knn_base: 16 GridSearchCV workers already use all cores.
# Adding 16 KNN threads per worker creates 256 threads on 16 cores = degraded perf.
# n_jobs=-1 in GridSearchCV: parallelises the 100 (combo x fold) fits across processes.
# return_train_score omitted (default=False): including it forces a 20K train-set
# prediction per fold in addition to the 5K validation prediction, causing 5x slowdown.
param_grid = {
    'n_neighbors': [3, 5, 7, 11, 15],
    'metric':      ['euclidean', 'manhattan'],
    'weights':     ['uniform', 'distance'],
}
log(f"\n[3] GridSearchCV -- {5*2*2} combos x 5 folds = 100 fits")
log(f"  n_jobs=-1 in GridSearchCV | n_jobs=1 in KNN base | no return_train_score")
log(f"  Est. time: ~4-6 min on 16-core CPU")
log(f"  Start: {time.strftime('%H:%M:%S')}")

knn_base = KNeighborsClassifier(algorithm='ball_tree', n_jobs=1)
grid_search = GridSearchCV(
    estimator=knn_base,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

t0 = time.time()
grid_search.fit(X_knn, y_knn)
elapsed = time.time() - t0

log(f"  GridSearchCV complete in {elapsed:.1f}s  ({time.strftime('%H:%M:%S')})")
log(f"  Best params : {grid_search.best_params_}")
log(f"  Best CV F1  : {grid_search.best_score_:.4f}")

# 4. All CV results table (top 5)
cv_results = pd.DataFrame(grid_search.cv_results_)
log("\n[4] Top 5 combinations by mean validation F1:")
cols = ['param_n_neighbors','param_metric','param_weights',
        'mean_test_score','std_test_score','rank_test_score']
top5 = cv_results[cols].sort_values('rank_test_score').head(5)
log(top5.to_string(index=False))

# 5. k vs. F1 validation curve figure
log("\n[5] Generating k vs. F1 validation curve ...")
k_values = [3, 5, 7, 11, 15]
combos = [('euclidean','uniform'),('euclidean','distance'),
          ('manhattan','uniform'),('manhattan','distance')]

fig, ax = plt.subplots(figsize=(9, 5))
for metric, weights in combos:
    mask = ((cv_results['param_metric'] == metric) &
            (cv_results['param_weights'] == weights))
    subset = cv_results[mask].sort_values('param_n_neighbors')
    f1_means = subset['mean_test_score'].values
    f1_stds  = subset['std_test_score'].values
    ax.plot(k_values, f1_means, marker='o', label=f'{metric} / {weights}')
    ax.fill_between(k_values, f1_means - f1_stds, f1_means + f1_stds, alpha=0.12)

best_k = grid_search.best_params_['n_neighbors']
ax.axvline(best_k, color='red', linestyle='--', linewidth=1.2, label=f'Best k={best_k}')
ax.set_xlabel('k (number of neighbors)', fontsize=12)
ax.set_ylabel('Mean 5-fold Validation F1', fontsize=12)
ax.set_title('k-NN: k vs. Validation F1 (GridSearchCV -- 25K subsample)', fontsize=13)
ax.xaxis.set_major_locator(mticker.FixedLocator(k_values))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/knn_k_vs_f1.png', dpi=150, bbox_inches='tight')
plt.close()
log("  Saved results/knn_k_vs_f1.png")

# 6. Evaluate best model on test set.
# Refit with n_jobs=-1 so the three large predict calls (36K test + 36K proba +
# 25K train) use all cores and complete in ~90 seconds instead of ~14 minutes.
log("\n[6] Fitting final model with n_jobs=-1 for fast evaluation ...")
best_params = grid_search.best_params_
final_knn = KNeighborsClassifier(algorithm='ball_tree', n_jobs=-1, **best_params)
final_knn.fit(X_knn, y_knn)
log("  Final model fitted.")

log("\n[6b] Predicting on test set (36K rows, n_jobs=-1) ...")
t0 = time.time()
y_pred  = final_knn.predict(X_test)
y_proba = final_knn.predict_proba(X_test)[:, 1]
log(f"  Test predictions complete in {time.time()-t0:.1f}s")

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_proba)

log("\n[6c] Computing train F1 on subsample ...")
t0 = time.time()
y_pred_train = final_knn.predict(X_knn)
train_f1     = f1_score(y_knn, y_pred_train)
log(f"  Train prediction complete in {time.time()-t0:.1f}s")

log(f"\n  === Test Set Metrics ===")
log(f"  Accuracy  : {accuracy:.4f}")
log(f"  Precision : {precision:.4f}")
log(f"  Recall    : {recall:.4f}")
log(f"  F1-Score  : {f1:.4f}")
log(f"  ROC-AUC   : {roc_auc:.4f}")
log(f"  Train F1  : {train_f1:.4f}  (on 25K subsample)")

# 7. Confusion matrix figure
log("\n[7] Generating confusion matrix ...")
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp_val = cm.ravel()
log(f"  TN:{tn:,}  FP:{fp:,}  FN:{fn:,}  TP:{tp_val:,}")

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['On-Time (0)', 'Late (1)'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('k-NN -- Confusion Matrix (Test Set)', fontsize=13)
plt.tight_layout()
plt.savefig('results/knn_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
log("  Saved results/knn_confusion_matrix.png")

# 8. 5-fold Stratified CV on the subsampled training set.
# cv_estimator uses n_jobs=1 to avoid thread contention with the 5 parallel CV workers.
log("\n[8] 5-fold Stratified CV on 25K subsample ...")
cv_estimator = KNeighborsClassifier(algorithm='ball_tree', n_jobs=1, **best_params)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
t0 = time.time()
cv_f1_scores = cross_val_score(cv_estimator, X_knn, y_knn,
                                cv=skf, scoring='f1', n_jobs=-1)
log(f"  CV complete in {time.time()-t0:.1f}s")
for i, s in enumerate(cv_f1_scores, 1):
    log(f"  Fold {i}: {s:.4f}")
log(f"  Mean: {cv_f1_scores.mean():.4f}  Std: {cv_f1_scores.std():.4f}")

# 9. Save knn_results.pkl (PHASES.md §2.3 schema)
log("\n[9] Saving results/knn_results.pkl ...")
knn_results = {
    'model_name':         'k-NN',
    'model':              final_knn,
    'best_params':        best_params,
    'y_pred':             y_pred,
    'y_proba':            y_proba,
    'metrics': {
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'roc_auc':   roc_auc,
    },
    'cv_f1_scores':       cv_f1_scores,
    'train_f1':           train_f1,
    'test_f1':            f1,
    'feature_importance': None,
}

required_keys = {'model_name','model','best_params','y_pred','y_proba',
                 'metrics','cv_f1_scores','train_f1','test_f1','feature_importance'}
assert set(knn_results.keys()) == required_keys, 'Schema mismatch!'
assert set(knn_results['metrics'].keys()) == {'accuracy','precision','recall','f1','roc_auc'}

with open('results/knn_results.pkl', 'wb') as f:
    pickle.dump(knn_results, f)

log("  DONE. knn_results.pkl saved.")
log(f"  Keys    : {list(knn_results.keys())}")
log(f"  Metrics : {knn_results['metrics']}")
log(f"\n=== Script complete at {time.strftime('%H:%M:%S')} ===")
