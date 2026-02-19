"""
Sample Submission — Active-Learning Talent Fraud Detector
==========================================================
Sponsored problem by Eightfold AI.

This is a reference implementation that demonstrates a competitive strategy.
You are free to use, modify, or completely replace it.

Strategy (two-stage active learning):
  Stage 1 — Explore  (up to 40 queries)
    Query rows that score highly on each of the four fraud dimensions:
    credential fraud signals, application bombing signals,
    account-takeover signals, and ghost-profile signals.
    Top-8 rows per dimension are unioned (deduplication may reduce this),
    plus up to 8 random rows to anchor the legitimate distribution.
    This seeds each fraud cluster with labeled examples.

  Stage 2 — Exploit  (remaining budget, up to ~60 queries)
    Train a Gradient Boosted tree on the Stage-1 labels.
    Query the rows with the highest predicted fraud probability that
    haven't been labeled yet.  This densifies coverage of the clusters
    already identified in Stage 1.

Final prediction:
    Retrain the model on all 100 labeled rows (balanced sample weights)
    and predict binary labels for every row in the dataset.

Expected performance:
    Random baseline  F1 ≈ 0.25–0.35
    This agent       F1 ≈ 0.40–0.55   (varies by random seed)
    Upper bound*     F1 ≈ 0.70–0.80   (*with perfect cluster coverage)

Required signature:
    run_agent(df, oracle_fn, budget) -> np.ndarray of int (0/1), length == len(df)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight


# ── helpers ──────────────────────────────────────────────────────────────────

def _anomaly_scores(df: pd.DataFrame) -> np.ndarray:
    """
    Four hand-crafted anomaly dimensions, one per fraud cluster.
    Each score is high when the row looks like that cluster's fraud type.
    No label information is used — these are computed from features only.
    """
    # C1: credential fraud (risky institution/company, suspicious GPA, large gaps)
    c1 = (df["institution_risk_score"]
          + df["gpa_anomaly_score"]
          + df["company_risk_score"]
          + df["tenure_gap_months"] / 20.0)

    # C2: application bombing (extreme velocity, near-zero idle time)
    c2 = (df["applications_7d"] / 10.0
          + df["applications_30d"] / 30.0
          + df["app_to_avg_ratio"]
          - df["time_since_last_app_hrs"])

    # C3: account takeover (new device + failed logins + high login velocity)
    c3 = (df["email_risk_score"]
          + df["is_new_device"]
          + df["failed_logins_24h"]
          + df["login_velocity_24h"] / 5.0)

    # C4: ghost profile (brand-new account + high copy-paste + skill inflation)
    c4 = (df["copy_paste_ratio"]
          + df["skills_to_exp_ratio"] / 10.0
          - df["profile_age_days"] / 500.0)

    return np.stack([c1.values, c2.values, c3.values, c4.values], axis=1)


def _fit_model(X_tr: np.ndarray, y_tr: np.ndarray, seed: int = 0):
    """
    Fit a classifier with balanced sample weights to handle class imbalance.
    Falls back to logistic regression when GBM requirements aren't met.
    """
    n_fraud = int(y_tr.sum())
    if n_fraud < 3 or n_fraud >= len(y_tr) - 1:
        clf = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0)
        clf.fit(X_tr, y_tr)
        return clf

    sw = compute_sample_weight("balanced", y_tr)
    gbm = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.1,
        random_state=seed,
    )
    gbm.fit(X_tr, y_tr, sample_weight=sw)
    return gbm


# ── main entry point ──────────────────────────────────────────────────────────

def run_agent(df: pd.DataFrame, oracle_fn, budget: int) -> np.ndarray:
    """
    Parameters
    ----------
    df        : pd.DataFrame, shape (n, n_features) — the full unlabelled dataset
    oracle_fn : callable(indices: list[int]) -> list[int]
                Returns binary labels (0/1) for the requested row indices.
                0 = legitimate candidate, 1 = fraudulent.
                Raises BudgetExceededError if you exceed `budget` total queries.
    budget    : int — maximum oracle calls allowed (typically 100)

    Returns
    -------
    predictions : np.ndarray of int, shape (n,) — 0 = legitimate, 1 = fraud
    """
    n = len(df)
    rng = np.random.default_rng(42)
    X_full = df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    scores = _anomaly_scores(df)  # shape (n, 4)

    labeled_idx = []   # indices we have queried
    labeled_y   = []   # corresponding labels

    # ── Stage 1: Explore — 8 rows per anomaly dimension + 8 diverse random ──
    explore_per_dim = min(8, budget // (4 + 1))
    stage1_idx: set[int] = set()

    for dim in range(4):
        top = np.argsort(-scores[:, dim])[:explore_per_dim]
        stage1_idx.update(top.tolist())

    # A few diverse random rows to anchor the legitimate distribution
    random_pool = [i for i in range(n) if i not in stage1_idx]
    n_random = min(8, budget - len(stage1_idx))
    stage1_idx.update(rng.choice(random_pool, size=n_random, replace=False).tolist())

    stage1_list = list(stage1_idx)
    labels_stage1 = oracle_fn(stage1_list)
    labeled_idx.extend(stage1_list)
    labeled_y.extend(labels_stage1)

    budget_used = len(stage1_list)

    # ── Stage 2: Exploit — query rows with highest predicted fraud probability ──
    remaining = budget - budget_used
    if remaining > 0 and sum(labeled_y) >= 2:
        y_arr = np.array(labeled_y)
        model1 = _fit_model(scaler.transform(X_full[labeled_idx]), y_arr, seed=0)
        proba1 = model1.predict_proba(X_scaled)[:, 1]

        not_queried = np.array([i for i in range(n) if i not in set(labeled_idx)])
        # Query the rows the model thinks are most likely fraudulent
        exploit_idx = not_queried[np.argsort(-proba1[not_queried])[:remaining]]
        labels_stage2 = oracle_fn(exploit_idx.tolist())
        labeled_idx.extend(exploit_idx.tolist())
        labeled_y.extend(labels_stage2)

    # ── Final model: retrain on all 100 labeled rows ─────────────────────────
    idx_arr = np.array(labeled_idx)
    y_arr   = np.array(labeled_y)

    final_model = _fit_model(scaler.transform(X_full[idx_arr]), y_arr, seed=1)
    predictions = final_model.predict(X_scaled).astype(int)

    return predictions
