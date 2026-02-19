"""
Local oracle for testing your agent before submission.

This is the SAME interface as the oracle used during evaluation.
The only difference is that the evaluation oracle runs against a
DIFFERENT dataset generated with a different random seed — so your
local score is a proxy, not the exact final score.

The labels for the local dataset are stored in labels.npy (same directory).
You may read labels.npy directly for debugging, but your agent.py must
only access labels through oracle_fn() — that is how the real evaluation works.

Usage inside your agent:
    labels = oracle_fn([0, 5, 99, 204, 7777])   # → list of ints: [0, 1, 0, 0, 1]
"""

import numpy as np
from pathlib import Path


class BudgetExceededError(Exception):
    """Raised when your agent requests more indices than the allowed budget."""


class Oracle:
    """
    Callable oracle that returns ground-truth labels for requested row indices.

    Parameters
    ----------
    budget : int
        Maximum total number of indices that may be queried across ALL calls.
        Default is 100 (the same limit used in evaluation).

    Examples
    --------
        oracle = Oracle()
        labels = oracle([0, 1, 2, 3, 4])   # costs 5 from budget
        labels = oracle(range(5, 10))       # costs 5 more

    Calling oracle([i1, i2, ...]) is identical to calling oracle_fn([i1, i2, ...])
    inside run_agent — they are the same object.
    """

    def __init__(self, budget: int = 100):
        labels_path = Path(__file__).parent / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"labels.npy not found at {labels_path}. "
                "Run: python ../dataset_gen/generate.py --split participant --output-dir ."
            )
        self._labels = np.load(labels_path).astype(int)
        self._budget = int(budget)
        self._used = 0

    def __call__(self, indices) -> list[int]:
        """
        Return labels for the given row indices.

        Parameters
        ----------
        indices : iterable of int
            Row numbers into dataset.csv (0-indexed). Any length is fine,
            but total indices across ALL calls must not exceed the budget.

        Returns
        -------
        list[int]  — 0 (legitimate) or 1 (fraud), same order as indices.

        Raises
        ------
        BudgetExceededError  if this call would exceed the remaining budget.
        IndexError           if any index is out of range.
        """
        try:
            indices = list(indices)
        except TypeError:
            raise TypeError("indices must be iterable (e.g. a Python list)")

        if not indices:
            return []

        n = len(indices)

        if self._used + n > self._budget:
            remaining = self._budget - self._used
            raise BudgetExceededError(
                f"Budget exceeded! "
                f"Budget={self._budget} | Used={self._used} | "
                f"Requested={n} | Remaining={remaining}. "
                "Reduce your query size or restructure your sampling strategy."
            )

        n_rows = len(self._labels)
        for idx in indices:
            if not isinstance(idx, (int, np.integer)):
                raise TypeError(
                    f"Index must be an integer, got {type(idx).__name__}: {idx!r}"
                )
            if not (0 <= int(idx) < n_rows):
                raise IndexError(
                    f"Index {idx} is out of range. "
                    f"Valid range: [0, {n_rows - 1}]."
                )

        self._used += n
        return [int(self._labels[int(i)]) for i in indices]

    @property
    def queries_used(self) -> int:
        """How many indices have been queried so far."""
        return self._used

    @property
    def budget(self) -> int:
        """Maximum total indices allowed."""
        return self._budget

    @property
    def budget_remaining(self) -> int:
        """How many more indices you may query."""
        return self._budget - self._used

    def reset(self) -> None:
        """Reset the budget counter. Useful for iterative local testing."""
        self._used = 0

    def __repr__(self) -> str:
        return (
            f"Oracle(budget={self._budget}, used={self._used}, "
            f"remaining={self.budget_remaining})"
        )
