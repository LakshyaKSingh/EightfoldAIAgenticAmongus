"""
Local testing framework — mirrors the exact evaluation harness used during grading.

Run this script to test your agent against the local dataset:

    python framework.py               # tests agent.py
    python framework.py --agent my_agent.py

Your agent will be called with:
    run_agent(df, oracle_fn, budget=100)

The score you see here is against the LOCAL dataset (dataset.csv).
The ACTUAL evaluation uses a structurally identical dataset generated
with a different random seed, so local and final scores will differ slightly.
"""

import time
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from oracle import Oracle, BudgetExceededError

BUDGET       = 100
DATASET_PATH = Path(__file__).parent / "dataset.csv"
AGENT_PATH   = Path(__file__).parent / "agent.py"


def _load_agent(path: Path):
    spec   = importlib.util.spec_from_file_location("agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coerce_predictions(raw, n: int) -> np.ndarray:
    if raw is None:
        return np.zeros(n, dtype=int)
    arr = np.asarray(raw, dtype=float).flatten()
    if len(arr) < n:
        print(f"  [framework] Warning: {len(arr)} predictions returned, expected {n}. "
              f"Padding {n - len(arr)} missing rows with 0.")
        arr = np.concatenate([arr, np.zeros(n - len(arr))])
    elif len(arr) > n:
        arr = arr[:n]
    return (arr >= 0.5).astype(int)


def run(agent_path: Path = AGENT_PATH) -> None:
    # ── Load dataset ──────────────────────────────────────────────────────────
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"dataset.csv not found at {DATASET_PATH}.\n"
            "Generate it with:\n"
            "  python ../dataset_gen/generate.py --split participant --output-dir ."
        )
    df = pd.read_csv(DATASET_PATH)
    n  = len(df)

    # ── Load agent ────────────────────────────────────────────────────────────
    if not agent_path.exists():
        raise FileNotFoundError(f"Agent not found: {agent_path}")

    print(f"Loading agent : {agent_path}")
    agent = _load_agent(agent_path)

    if not hasattr(agent, "run_agent"):
        raise AttributeError(
            "Your agent file must define a function named 'run_agent'. "
            "See example_agent.py for the required signature."
        )

    oracle = Oracle(budget=BUDGET)

    # ── Run ───────────────────────────────────────────────────────────────────
    print(f"Running agent with budget={BUDGET} on {n} rows ...\n")
    start = time.perf_counter()
    raw_preds = None

    try:
        raw_preds = agent.run_agent(df.copy(), oracle, BUDGET)

    except BudgetExceededError as e:
        elapsed = time.perf_counter() - start
        print(f"\n  BudgetExceededError: {e}")
        print(f"  Queries used before error: {oracle.queries_used}")
        print(f"  Elapsed: {elapsed:.1f}s")
        print("\n  Scoring whatever predictions were returned (if any) ...")

    except Exception as e:
        elapsed = time.perf_counter() - start
        import traceback
        print(f"\n  Agent crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"\n  Queries used before crash: {oracle.queries_used}")
        print(f"  Elapsed: {elapsed:.1f}s")
        print("\n  Scoring whatever predictions were returned (if any) ...")

    elapsed = time.perf_counter() - start

    # ── Score ─────────────────────────────────────────────────────────────────
    preds  = _coerce_predictions(raw_preds, n)
    labels = np.load(Path(__file__).parent / "labels.npy").astype(int)

    f1   = f1_score(labels,   preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels,  preds, zero_division=0)
    cm   = confusion_matrix(labels, preds)

    # ── Report ────────────────────────────────────────────────────────────────
    sep = "─" * 48
    print(sep)
    print(f"  F1 Score      : {f1:.4f}   ← primary metric")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall        : {rec:.4f}")
    print(sep)
    print(f"  Queries used  : {oracle.queries_used} / {BUDGET}")
    print(f"  Runtime       : {elapsed:.2f}s")
    print(sep)
    print(f"  Confusion matrix (rows=actual, cols=predicted):")
    print(f"            Pred 0  Pred 1")
    print(f"  Actual 0  {cm[0,0]:>6}  {cm[0,1]:>6}")
    print(f"  Actual 1  {cm[1,0]:>6}  {cm[1,1]:>6}")
    print(sep)

    # Tie-breaker reminder
    print("  Tie-breaking order used in final evaluation:")
    print("    1. F1 score    (higher is better)")
    print("    2. Queries used (fewer is better)")
    print("    3. Runtime     (faster is better)")
    print(sep)
    print("  Note: final evaluation uses a DIFFERENT dataset (same structure,")
    print("        different random seed). Local score ≈ final score but won't match exactly.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test your agent locally.")
    parser.add_argument(
        "--agent", type=str, default=str(AGENT_PATH),
        help="Path to the Python file containing run_agent() (default: agent.py)",
    )
    args = parser.parse_args()
    run(Path(args.agent))
