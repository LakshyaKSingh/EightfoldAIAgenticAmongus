# Innov8 4.0 — Talent Fraud Detection Challenge
### Sponsored by [Eightfold AI](https://eightfold.ai)

## The Problem: Active Learning with a Limited Oracle

Eightfold AI's talent intelligence platform processes millions of candidate
profiles and job applications globally. A small fraction of these are
**fraudulent** — fake credentials, inflated experience, bot-driven spam
applications, and hijacked accounts.

You are given a dataset of **10,000 candidate profiles** (features only, no labels).
Your task is to build a classifier that identifies which profiles are fraudulent.

**Constraint:** you may query the ground-truth label for at most **100 rows**.
Every label query costs one unit of budget. How you allocate that budget determines your score.

---

## Dataset

`dataset.csv` — 10,000 rows × 25 feature columns. No label column is included.

| Feature | Type | Description |
|---|---|---|
| `profile_age_days` | int | Days since the candidate profile was created |
| `applications_7d` | int | Number of job applications submitted in the last 7 days |
| `applications_30d` | int | Number of job applications submitted in the last 30 days |
| `avg_applications_30d` | float | Historical average daily applications over the last 30 days |
| `app_to_avg_ratio` | float | `applications_30d / (avg_applications_30d × 30)` |
| `skills_count` | int | Number of skills listed on the profile |
| `endorsements_count` | int | Number of skill endorsements received |
| `experience_years` | float | Total claimed years of work experience |
| `skills_to_exp_ratio` | float | `skills_count / (experience_years + 1)` |
| `institution_risk_score` | float 0–1 | Risk score for the listed educational institution(s) |
| `company_risk_score` | float 0–1 | Risk score for the listed employer(s) |
| `gpa_anomaly_score` | float 0–1 | How anomalous the claimed GPA is vs. the institution (0 = plausible, 1 = very suspicious) |
| `tenure_gap_months` | float | Total unexplained employment gaps (months) |
| `avg_tenure_months` | float | Average tenure per job role |
| `time_since_last_app_hrs` | float | Hours elapsed since the previous job application |
| `is_new_device` | int 0/1 | 1 if the login device was not previously seen for this account |
| `ip_risk_score` | float 0–1 | Risk score of the originating IP address |
| `email_risk_score` | float 0–1 | Risk score of the account email address |
| `login_velocity_24h` | int | Number of logins in the last 24 hours |
| `failed_logins_24h` | int 0–5 | Failed login attempts in the last 24 hours |
| `copy_paste_ratio` | float 0–1 | Fraction of profile text that appears copy-pasted from other profiles |
| `feature_noise_1–4` | mixed | Uninformative noise features |

Overall fraud rate: **approximately 8%** (≈800 fraudulent profiles out of 10,000).

### Fraud types in the dataset

| Type | Description | Key signals |
|---|---|---|
| **Credential Fraud** | Fake degrees, ghost companies, implausible GPAs | `institution_risk_score`, `gpa_anomaly_score`, `company_risk_score`, `tenure_gap_months` |
| **Application Bombing** | Bots mass-applying to every open role | `applications_7d`, `applications_30d`, `app_to_avg_ratio`, `time_since_last_app_hrs` |
| **Account Takeover** | Attacker hijacks a legitimate profile | `is_new_device`, `failed_logins_24h`, `login_velocity_24h`, `email_risk_score` |
| **Ghost Profile** | Freshly fabricated synthetic identity | `profile_age_days`, `copy_paste_ratio`, `skills_to_exp_ratio` |

---

## Your Task

Implement the `run_agent` function in a file named `agent.py` (or any name declared in `manifest.json`):

```python
import numpy as np
import pandas as pd

def run_agent(df: pd.DataFrame, oracle_fn, budget: int) -> np.ndarray:
    """
    Parameters
    ----------
    df        : pd.DataFrame, shape (10000, 25)
                The full feature matrix. Row indices 0–9999.
                No label column is present.

    oracle_fn : callable
                oracle_fn(indices: list[int]) -> list[int]
                    indices  — list of row indices to query (0-based)
                    returns  — list of int labels, same length, values in {0, 1}
                               0 = legitimate candidate, 1 = fraudulent

                Raises BudgetExceededError if the TOTAL number of indices
                across ALL calls to oracle_fn exceeds `budget`.
                BudgetExceededError is a subclass of Exception.

    budget    : int — maximum oracle queries available (100 in final evaluation)

    Returns
    -------
    predictions : np.ndarray, dtype int, shape (10000,)
                  Your binary predictions for EVERY row in `df`.
                  Values must be in {0, 1}.
    """
```

### Oracle calling examples

```python
# Single call for multiple rows
labels = oracle_fn([0, 42, 137, 9999])          # returns [0, 1, 0, 0]

# Multiple calls (all count toward the same budget)
labels_a = oracle_fn([0, 1, 2])                  # 3 queries used
labels_b = oracle_fn(list(range(3, 100)))        # 97 more queries
# budget now exhausted; any further call raises BudgetExceededError
```

---

## Scoring

### Primary metric: F1 score (fraud class)

```
F1 = 2 × Precision × Recall / (Precision + Recall)
```

Evaluated on a **held-out dataset** that has the same 25 features and similar statistics as `dataset.csv`, but different rows. Your local F1 score (on `dataset.csv`) is an estimate, not the final score.

### Partial predictions are accepted

If your function returns fewer than 10,000 predictions, the remaining positions are treated as **0 (legitimate)**. If your function raises an exception or is killed by the timeout, predictions returned so far are used (padded to 10,000 with 0 if needed).

You may return float probabilities — values ≥ 0.5 are treated as fraud (1), values < 0.5 as legitimate (0).

### Tie-breaking (applied in this order when F1 scores are equal)

| Priority | Criterion | Direction |
|---|---|---|
| 1st | F1 score | Higher is better |
| 2nd | Oracle queries used | **Fewer is better** |
| 3rd | Wall-clock runtime (seconds) | **Faster is better** |

Using fewer oracle queries for the same accuracy is rewarded. Two agents with identical F1 are ranked by how many of their 100 queries they actually used.

---

## Evaluation Pipeline

When you submit, the evaluator performs the following steps:

1. **Security scan** — all `.py` files in your ZIP are parsed with `ast`. Forbidden imports (`os`, `sys`, `subprocess`, `socket`, `requests`, `urllib`, `pickle`, `gc`, `ctypes`, etc.) and forbidden calls (`eval`, `exec`, `open`, `__import__`, etc.) cause immediate **DISQUALIFIED** status.

2. **Import check** — your code may only import from the allowlist:
   `numpy`, `pandas`, `sklearn`, `scipy`, `math`, `random`, `statistics`,
   `collections`, `itertools`, `functools`, `typing`, `warnings`, `copy`, `time`, `json`, `re`

3. **Execution** — the evaluator loads `eval_features.csv` (hidden), creates an `Oracle` with budget=100, then calls:
   ```python
   predictions = run_agent(eval_df, oracle, 100)
   ```

4. **Timeout** — your agent has **5 minutes (300 seconds)** of wall-clock time. If it exceeds this, it is killed and the current predictions (if any) are scored.

5. **Budget enforcement** — calling `oracle_fn` with more indices than the remaining budget raises `BudgetExceededError`. You are responsible for tracking your usage via `oracle.queries_used` and `oracle.budget_remaining`.

6. **Scoring** — `F1`, `precision`, `recall`, `queries_used`, and `runtime_seconds` are recorded. Submissions with status ERROR, TIMEOUT, or BUDGET_EXCEEDED are still scored on whatever predictions were produced (which may give a very low F1 if the function crashed early).

7. **Ranking** — all submissions are ranked by (−F1, queries_used, runtime_seconds).

### Submission statuses

| Status | Meaning |
|---|---|
| `OK` | Completed normally; scored |
| `ERROR` | Exception before returning; partial predictions scored |
| `TIMEOUT` | Exceeded 5-minute wall clock; partial predictions scored |
| `BUDGET_EXCEEDED` | Called oracle beyond budget limit; partial predictions scored |
| `DISQUALIFIED` | Forbidden import or banned function found by AST scan; score = 0 |
| `INTERNAL_ERROR` | Evaluator bug; no score assigned |

---

## Submission Format

Your submission must be a **ZIP file** with the following structure:

```
your_team_submission.zip
├── manifest.json          ← required, must be at the root level
├── agent.py               ← or the filename in manifest["entry_point"]
├── helpers.py             ← optional helper modules
└── requirements.txt       ← optional (only from the allowed list)
```

### manifest.json

```json
{
  "team_name":   "Your Team Name",
  "team_id":     "your_unstop_id",
  "institution": "Your College / Company",
  "members": [
    {"name": "Alice Smith", "email": "alice@example.edu", "role": "lead"},
    {"name": "Bob Jones",   "email": "bob@example.edu",   "role": "member"}
  ],
  "entry_point": "agent.py"
}
```

**Required fields:** `team_name`, `team_id`, `institution`, `members`, `entry_point`.

- `team_id` must match your Unstop registration exactly.
- `entry_point` is the filename of the Python file containing `run_agent`. It may not include a directory prefix.
- `members` is a list of objects; each object must have at least a `"name"` key.
- The ZIP may contain nested directories; `manifest.json` is found by recursive search. **Path-traversal filenames (e.g., `../../etc/passwd`) cause immediate rejection.**

### Creating the ZIP

```bash
cd your_submission_folder/
bash package.sh YOURTEAMID
# produces YOURTEAMID_submission.zip
```

Or manually:

```bash
zip -r YOURTEAMID_submission.zip manifest.json agent.py helpers.py
```

---

## Allowed Libraries

```
numpy        pandas       scikit-learn (sklearn)   scipy
math         random       statistics               collections
itertools    functools    typing                   warnings
copy         time         json                     re
```

Any import outside this list causes **DISQUALIFIED** status. This includes:
`os`, `sys`, `subprocess`, `socket`, `requests`, `urllib`, `pickle`,
`gc`, `inspect`, `ctypes`, `importlib`, `threading`, `multiprocessing`.

---

## Local Testing

```bash
pip install -r requirements.txt

# Test the reference sample agent
python framework.py --agent sample_submission/agent.py

# Test your own agent
python framework.py --agent agent.py
```

`framework.py` uses `dataset.csv` + `labels.npy` (available to you) to simulate the evaluation. The output includes F1, precision, recall, queries used, runtime, and the tie-breaking reminder.

The **final evaluation uses a different dataset** (same features, different random seed). A gap between local and final scores is expected and normal.

---

## Rules

1. Do not hardcode information about the hidden evaluation dataset.
2. Your `run_agent` must not catch and suppress `BudgetExceededError` after budget is exhausted — doing so has no effect since the oracle still stops responding.
3. One submission per team. The last submission received before the deadline is evaluated.
4. No external network calls, file I/O, or subprocess creation. These are blocked by the AST scan.
5. You may NOT include data files (`.csv`, `.npy`) in your submission ZIP — they are ignored.

---

## Strategy Tips

- **Random sampling gives ~8 fraud in 100 queries.** That's the same rate as the overall dataset.
  A smart strategy can yield 30–50 fraud in 100 queries — a 4–6× improvement in labeled fraud density.

- **The fraud data has cluster structure.** Four distinct fraud types leave different feature fingerprints:
  credential fraud, application bombing, account takeover, and ghost profiles.
  Spending your budget to find these clusters is far more valuable than sampling uniformly.

- **Explore then exploit.** Start with diverse queries to discover fraud patterns,
  then use a trained model to zoom in on high-confidence fraud regions.

- **Class imbalance matters.** With only ~8 positive labels from random sampling,
  use `class_weight='balanced'`, oversampling, or `compute_sample_weight('balanced', y)`.

- **Efficiency is rewarded.** If two teams achieve the same F1, the one that used fewer oracle queries ranks higher.

- **Partial predictions are safe.** If your code might time out, emit predictions before the deadline
  by returning early (the framework will pad zeros for unset positions).

---

## Files

| File | Purpose |
|---|---|
| `dataset.csv` | 10,000-row feature matrix for local testing (no labels) |
| `labels.npy` | Local labels — **for testing only, do not submit** |
| `oracle.py` | Oracle with budget enforcement — identical interface to grading |
| `framework.py` | Local test harness — runs your agent and prints the score |
| `example_agent.py` | Baseline agent (random sampling + logistic regression) |
| `sample_submission/agent.py` | Reference smart agent (two-stage active learning) |
| `sample_submission/manifest.json` | Example manifest |
| `sample_submission/package.sh` | Script to create the submission ZIP |
| `requirements.txt` | Python dependencies |
| `manifest_template.json` | Manifest template to fill in |
