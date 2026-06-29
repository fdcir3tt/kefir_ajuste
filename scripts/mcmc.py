import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

from kefir_ajuste.equations import gompertz_eq_solution,gompertz_latent_eq
from kefir_ajuste.data import load_data,load_time_domain
# -----------------------------------------------------------------------------
# 1. SETUP SYNTHETIC DATA (True Parameters)
# -----------------------------------------------------------------------------
np.random.seed(42)

# True physical system parameters
DATA_FILE_NAME = "control_dataset.csv"
TRUE_R         = 0.046                   # Intrinsic growth rate
TRUE_K         = 47.81                   # Carrying capacity
TRUE_Y0        = 14.326143333333334      # Initial population size
TRUE_SIGMA     = 0.15                    # Multiplicative log-normal noise scale
ODE            = "gompertz"


config_dict  = {"gompertz":(gompertz_eq_solution,gompertz_latent_eq)}
ode_solution,latent_eq_solution = config_dict[ODE]

parameter_names = ['r', 'K', 'Y0', 'sigma']
true_parameter_values = {
    'r': TRUE_R,
    'K': TRUE_K,
    'Y0': TRUE_Y0,
    'sigma': TRUE_SIGMA,
}

initial_prior_conditions = {
    'r' : 0.2,
    'K' : 450,
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



# -----------------------------------------------------------------------------
#  Load data
# -----------------------------------------------------------------------------

dataset = load_data(DATA_FILE_NAME)
dataset = dataset.drop(columns=["tratamiento","Unnamed: 0"])
t0,tf   = load_time_domain(dataset)
t_data  = np.linspace(t0, tf, 200)

# Generate clean curve and inject log-normal noise
y_clean = ode_solution(t_data,true_parameter_values)
y_obs   = y_clean * np.random.lognormal(mean=0.0, 
                                        sigma=TRUE_SIGMA, 
                                        size=len(t_data))

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

# ArviZ changed summary interval APIs/column names across versions.
# Build a summary that works on both old and new releases.
try:
    summary = az.summary(trace, var_names=parameter_names, hdi_prob=0.94)
except TypeError:
    summary = az.summary(trace, var_names=parameter_names, ci_prob=0.94, ci_kind='hdi')

interval_columns = []

# Legacy ArviZ column names (for example: hdi_3%, hdi_97%)
legacy_hdi = ['hdi_3%', 'hdi_97%']
if set(legacy_hdi).issubset(summary.columns):
    interval_columns = legacy_hdi
else:
    # Modern ArviZ column names (for example: hdi94_lb, hdi94_ub)
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
plt.show()

# Extract posterior sample arrays for plotting
posterior = trace.posterior
samples_dict = {}
for param in posterior.keys():
    if param in initial_prior_conditions.keys():
        samples_dict[param] = (posterior[param].values
                                               .flatten())
        
r_samples = samples_dict.get('r')



plt.figure(figsize=(10, 6))
plt.scatter(t_data, y_obs, color='black', zorder=5, label='Observed Noisy Data')

# Draw 100 random trajectories from the posterior to show estimation uncertainty
t_plot = np.linspace(0, t_data.max(), 300)
for i in np.random.choice(len(r_samples), size=100, replace=False):
    parameters = {}
    for param in samples_dict.keys():
        parameters[param]=samples_dict.get(param)[i]
    plt.plot(t_plot, ode_solution(t_plot,parameters),
             color='cyan', alpha=0.1, zorder=1)

# Overlay the true underlying system curve
plt.plot(t_plot, ode_solution(t_plot, true_parameter_values),
         color='red', linestyle='--', linewidth=2, zorder=4, label='True System Curve')

# Visual configurations
plt.title("Bayesian MCMC Estimation of Gompertz ODE", fontsize=14)
plt.xlabel("Time ($t$)", fontsize=12)
plt.ylabel("Population / Size ($Y$)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()
