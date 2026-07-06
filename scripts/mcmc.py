import os
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

from pathlib import Path
from kefir_ajuste.equations import *
from kefir_ajuste.data import load_data,load_time_domain
# -----------------------------------------------------------------------------
# 1. SETUP  (True Parameters)
# -----------------------------------------------------------------------------
np.random.seed(42)

# True physical system parameters
DATA_FILE_NAME = "control_dataset.csv"
TRUE_R         = 0.046                   # Intrinsic growth rate
TRUE_K         = 47.81                   # Carrying capacity
TRUE_Y0        = 14.326143333333334      # Initial population size
TRUE_SIGMA     = 0.15                    # Multiplicative log-normal noise scale
ODE            = "verhulst"
TREATMENT      = 4

treatments = ["Testigo (T1) Kéfir sin ultrasonicar",
              "15 seg. 20 W/cm2 (T2)",
              "1 min. 20 W/cm2 (T3)",
              "15 seg. 34 W/cm2 (T4)",
              "1 min. 34 W/cm2 (T5)"]
save_path = Path("figures")
config_dict  = {"gompertz":(gompertz_eq_solution,gompertz_latent_eq),
                "verhulst":(verhulst_eq_solution,verhulst_latent_eq)}
ode_solution,latent_eq_solution = config_dict[ODE]

parameter_names = ['r', 'K', 'Y0', 'sigma']
true_parameter_values = {
    'r': TRUE_R,
    'K': TRUE_K,
    'Y0': TRUE_Y0,
    'sigma': TRUE_SIGMA,
}

initial_prior_conditions = {
    'r' : 0.1,
    'K' : 50,
    'Y0': 10,
}
prior_sigmas = {
    'r' : 0.5,
    'K' : 0.3,
    'Y0': 0.5,
    'sigma': 0.3,
}

def highest_density_interval(samples, prob=0.94):
    """Compute a 1D highest density interval using only NumPy."""
    values = np.asarray(samples, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("Cannot compute HDI on an empty sample array.")
    if not 0 < prob < 1:
        raise ValueError("prob must be between 0 and 1.")
    if values.size == 1:
        return values[0], values[0]

    sorted_values = np.sort(values)
    interval_size = int(np.ceil(prob * sorted_values.size))
    interval_size = min(max(interval_size, 2), sorted_values.size)

    widths = sorted_values[interval_size - 1:] - sorted_values[:sorted_values.size - interval_size + 1]
    min_width_idx = int(np.argmin(widths))
    hdi_lower = sorted_values[min_width_idx]
    hdi_upper = sorted_values[min_width_idx + interval_size - 1]
    return hdi_lower, hdi_upper


def plot_posterior_fallback(trace, var_names, true_values, hdi_prob=0.94, figsize=(12, 8)):
    """Version-stable posterior plot if ArviZ plotting API is unavailable."""
    n_params = len(var_names)
    n_cols = 2
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat_axes = axes.ravel()
    posterior = trace.posterior

    for i, parameter_name in enumerate(var_names):
        ax = flat_axes[i]
        if parameter_name not in posterior:
            ax.set_visible(False)
            continue

        samples = np.asarray(posterior[parameter_name].values).ravel()
        hdi_lower, hdi_upper = highest_density_interval(samples, prob=hdi_prob)
        posterior_mean = float(np.mean(samples))
        true_value = true_values.get(parameter_name)

        ax.hist(samples, bins="auto", density=True, color="tab:blue", alpha=0.7, edgecolor="white")
        ax.axvline(posterior_mean, color="black", linewidth=1.5, label="Posterior mean")
        ax.axvline(hdi_lower, color="tab:orange", linestyle="--", linewidth=1.2, label=f"{int(hdi_prob * 100)}% HDI")
        ax.axvline(hdi_upper, color="tab:orange", linestyle="--", linewidth=1.2)
        if true_value is not None:
            ax.axvline(true_value, color="red", linestyle=":", linewidth=1.8, label="True value")

        ax.set_title(f"Posterior of {parameter_name}")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="best", fontsize=8)

    for j in range(n_params, len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig.suptitle("Posterior Distributions by Parameter", fontsize=14)
    fig.tight_layout()
    return axes


os.makedirs(save_path/ODE/str(TREATMENT),exist_ok=True)

# -----------------------------------------------------------------------------
#  Load data
# -----------------------------------------------------------------------------
treatment = treatments[TREATMENT]
dataset = load_data(DATA_FILE_NAME)
dataset = dataset.drop(columns=["Unnamed: 0"])

df = dataset[dataset["tratamiento"]==treatment]
t0,tf   = load_time_domain(dataset)
t_data  = np.linspace(t0, tf, len(df))

# Generate clean curve and inject log-normal noise
y_clean = ode_solution(t_data,true_parameter_values)
y_obs   = df["concentracion(g/cm3)"]

# -----------------------------------------------------------------------------
# 2. BAYESIAN MODEL SPECIFICATION
# -----------------------------------------------------------------------------
with pm.Model() as gompertz_model:
    priors = {}
    for param in true_parameter_values.keys():
        if param =="sigma":
            sigma = prior_sigmas.get(param)
            priors[param]= pm.HalfNormal(param, 
                                         sigma=sigma)
            continue
        sigma = prior_sigmas.get(param)
        initial_condition = initial_prior_conditions.get(param)
        priors[param] = pm.LogNormal(param,
                                     mu=np.log(initial_condition),
                                     sigma=sigma)

    # Deterministic node embedding the analytical solution of the ODE
    # This maps parameters directly to expected trajectory paths at t_data
    y_latent = latent_eq_solution(t_data,priors)

    # Likelihood function: Log-Normal observation error model
    # Log-transforming both sides yields standard normal error: ln(y_obs) ~ N(ln(y_latent), sigma^2)
    likelihood = pm.LogNormal('y_obs', mu=pm.math.log(y_latent), sigma=sigma, observed=y_obs)


    # -------------------------------------------------------------------------
    # 3. RUN MCMC SAMPLING
    # -------------------------------------------------------------------------
    print("Running MCMC Sampler...")
    trace = pm.sample(draws=10000, tune=1000, target_accept=0.95, return_inferencedata=True)


# -----------------------------------------------------------------------------
# 4. RESULTS ANALYSIS
# -----------------------------------------------------------------------------

# Print text summary of parameter posteriors and convergence metrics (R-hat)
print("\n--- Posterior Parameter Summary ---")

try:
    summary = az.summary(trace, var_names=parameter_names, hdi_prob=0.94)
except TypeError:
    summary = az.summary(trace, var_names=parameter_names, ci_prob=0.94, ci_kind='hdi')

interval_columns = []
legacy_hdi = ['hdi_3%', 'hdi_97%']
if set(legacy_hdi).issubset(summary.columns):
    interval_columns = legacy_hdi
else:
    modern_hdi = [
        c for c in summary.columns
        if c.startswith('hdi') and (c.endswith('_lb') or c.endswith('_ub'))
    ]
    if len(modern_hdi) == 2:
        interval_columns = sorted(modern_hdi, key=lambda c: c.endswith('_ub'))

columns_to_print = ['mean', 'sd', *interval_columns, 'r_hat']
columns_to_print = [c for c in columns_to_print if c in summary.columns]
print(summary[columns_to_print])

# Posterior distributions for each parameter
if hasattr(az, "plot_posterior"):
    posterior_axes = az.plot_posterior(
        trace,
        var_names=parameter_names,
        hdi_prob=0.94,
        ref_val=true_parameter_values,
        figsize=(12, 8),
    )
    if hasattr(posterior_axes, "ravel"):
        for ax, parameter_name in zip(posterior_axes.ravel(), parameter_names):
            ax.set_title(f"Posterior of {parameter_name}")
    plt.suptitle("Posterior Distributions by Parameter", fontsize=14)
    plt.tight_layout()
else:
    plot_posterior_fallback(
        trace,
        var_names=parameter_names,
        true_values=true_parameter_values,
        hdi_prob=0.94,
        figsize=(12, 8),
    )
plt.savefig(save_path/ODE/str(TREATMENT)/"posterior_distributions.png")

# -----------------------------------------------------------------------------
# 4b. CREDIBLE BANDS (latent + predictive)
# -----------------------------------------------------------------------------

# Extract posterior sample arrays for ALL parameters (incluyendo sigma)
posterior = trace.posterior
samples_dict = {}
for param in posterior.keys():
    if param in true_parameter_values.keys():   # r, K, Y0, sigma
        samples_dict[param] = posterior[param].values.flatten()

r_samples = samples_dict.get('r')
sigma_samples = samples_dict.get('sigma')

n_samples_total = len(r_samples)

# Subsampling razonable si hay demasiadas muestras (ajusta según tu paciencia/CPU)
MAX_TRAJECTORIES = 3000
if n_samples_total > MAX_TRAJECTORIES:
    idx = np.random.choice(n_samples_total, size=MAX_TRAJECTORIES, replace=False)
else:
    idx = np.arange(n_samples_total)

t_plot = np.linspace(0, t_data.max(), 300)

latent_trajectories = np.zeros((len(idx), len(t_plot)))
predictive_trajectories = np.zeros((len(idx), len(t_plot)))

for row, i in enumerate(idx):
    parameters = {param: samples_dict[param][i] for param in ['r', 'K', 'Y0']}
    mean_traj = ode_solution(t_plot, parameters)
    latent_trajectories[row, :] = mean_traj

    sigma_i = sigma_samples[i]
    predictive_trajectories[row, :] = mean_traj * np.random.lognormal(
        mean=0.0, sigma=sigma_i, size=len(t_plot)
    )

HDI_PROB = 0.94
lower_q = (1 - HDI_PROB) / 2 * 100   # e.g. 3
upper_q = 100 - lower_q              # e.g. 97

# Banda latente (incertidumbre en los parámetros)
latent_lower  = np.percentile(latent_trajectories, lower_q, axis=0)
latent_upper  = np.percentile(latent_trajectories, upper_q, axis=0)
latent_median = np.percentile(latent_trajectories, 50, axis=0)

# Banda predictiva (incertidumbre en parámetros + ruido de observación)
pred_lower = np.percentile(predictive_trajectories, lower_q, axis=0)
pred_upper = np.percentile(predictive_trajectories, upper_q, axis=0)

# -----------------------------------------------------------------------------
# 4c. PLOT: Data + credible bands + true curve
# -----------------------------------------------------------------------------

plt.figure(figsize=(10, 6))

# Banda predictiva (más ancha, incluye ruido) - dibujar primero, atrás
plt.fill_between(t_plot, pred_lower, pred_upper,
                  color='gray', alpha=0.25, zorder=1,
                  label=f'{int(HDI_PROB*100)}% Predictive Band')

# Banda latente (más angosta, solo incertidumbre de parámetros)
plt.fill_between(t_plot, latent_lower, latent_upper,
                  color='cyan', alpha=0.4, zorder=2,
                  label=f'{int(HDI_PROB*100)}% Credibility Band(latent)')

plt.plot(t_plot, latent_median, color='blue', linewidth=1.5, zorder=3,
         label='Posterior Median')

plt.scatter(t_data, y_obs, color='black', zorder=5, s=20, label='Observed Data')

plt.plot(t_plot, ode_solution(t_plot, true_parameter_values),
         color='red', linestyle='--', linewidth=2, zorder=4, label='True System Curve')

plt.title(f"Bayesian MCMC Estimation of {ODE.capitalize()} ODE", fontsize=14)
plt.xlabel("Time ($t$)", fontsize=12)
plt.ylabel("Population / Size ($Y$)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.savefig(save_path/ODE/str(TREATMENT)/"plot.png")