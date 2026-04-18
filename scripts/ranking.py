import mlflow
import pandas as pd

# Set tracking URI if needed
# mlflow.set_tracking_uri("http://localhost:5000")

EXPERIMENT_NAME = "verhulst_multi_polynomial_random_collocation_discovery"

# Metrics to evaluate
METRICS = ["aic", "bic", "mape", "mae", "rmse"]

# Load experiment
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")

# Get runs
runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    output_format="pandas"
)

if runs_df.empty:
    raise ValueError("No runs found.")

# Keep only relevant columns
metric_cols = [f"metrics.{m}" for m in METRICS]
cols = ["run_id"] + metric_cols

df = runs_df[cols].copy()

# Drop runs missing any metric
df = df.dropna()

# Rename columns for convenience
df.columns = ["run_id"] + METRICS

# ---- OPTION 1: Single composite score (normalized) ----
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Lower is better for all your metrics
for m in METRICS:
    df[f"{m}_norm"] = normalize(df[m])

# Equal weighting (you can change this)
weights = {
    "aic": 1,
    "bic": 1,
    "mape": 1,
    "mae": 1,
    "rmse": 1,
}

df["score"] = sum(df[f"{m}_norm"] * weights[m] for m in METRICS)

# Lower score = better
df = df.sort_values("score")

# ---- OPTION 2: Pareto front (multi-objective) ----
def is_pareto_efficient(costs):
    n_points = costs.shape[0]
    is_efficient = [True] * n_points
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient = [
                e and not all(costs[j] <= c) or all(costs[j] == c)
                for j, e in enumerate(is_efficient)
            ]
            is_efficient[i] = True
    return is_efficient

pareto_mask = is_pareto_efficient(df[METRICS].values)
pareto_df = df[pareto_mask]

# ---- OUTPUT ----
print("\nTop 10 runs (composite score):")
print(df.head(10))

print("\nPareto-optimal runs:")
print(pareto_df)