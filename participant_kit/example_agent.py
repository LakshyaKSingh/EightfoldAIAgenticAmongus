"""
Example baseline agent — random sampling + logistic regression.

This is provided as a reference implementation to show the required interface.
It is a weak baseline. You are expected to do significantly better.

Expected F1 on local dataset: ~0.30 – 0.45

To test this agent:
    python framework.py --agent example_agent.py

To use as a starting point, copy this file to agent.py and improve it.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def run_agent(df: pd.DataFrame, oracle_fn, budget: int) -> np.ndarray:
    """
    Active learning agent entry point.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset — 10,000 rows of candidate profile features, NO labels.
        Do not add a label column to this DataFrame.

    oracle_fn : callable
        Call oracle_fn(list_of_indices) to get ground-truth labels.

        Examples:
            labels = oracle_fn([0, 5, 99])        # → [0, 1, 0]
            label  = oracle_fn([42])               # → [1]

        Rules:
          - Total indices across ALL calls must not exceed `budget`.
          - Exceeding the budget raises BudgetExceededError.
          - Indices must be integers in range [0, len(df)).

    budget : int
        Maximum total number of row indices you may query (100).

    Returns
    -------
    np.ndarray of shape (len(df),)
        Predicted labels — 0 (legitimate candidate) or 1 (fraudulent) — for every row.
        If you return fewer than len(df) predictions, remaining rows are
        treated as 0. Float probabilities are accepted (threshold = 0.5).
    """
    n = len(df)

    # ── Step 1: choose which rows to label ───────────────────────────────────
    # Baseline: pick randomly. A better strategy would use the feature space
    # structure to identify diverse or high-information regions — for example,
    # targeting rows with high institution_risk_score, extreme application
    # velocity, or suspicious copy_paste_ratio.
    query_indices = np.random.choice(n, size=budget, replace=False).tolist()

    # ── Step 2: query the oracle ──────────────────────────────────────────────
    # You can call oracle_fn with all indices at once or in smaller batches.
    # The total count across all calls must not exceed `budget`.
    labels = oracle_fn(query_indices)

    # ── Step 3: train a classifier on the labeled subset ─────────────────────
    X_train = df.iloc[query_indices].values
    y_train = np.array(labels, dtype=int)

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # important: fraud is rare (~8%)
        C=1.0,
        solver="lbfgs",
    )
    clf.fit(X_train_sc, y_train)

    # ── Step 4: predict on all 10,000 rows ───────────────────────────────────
    X_all_sc = scaler.transform(df.values)
    predictions = clf.predict(X_all_sc)

    return predictions
