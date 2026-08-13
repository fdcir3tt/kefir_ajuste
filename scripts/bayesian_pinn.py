import os
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt

from pathlib import Path
from kefir_ajuste.equations import *
from kefir_ajuste.data import load_data, load_time_domain

# -----------------------------------------------------------------------------
# 1. SETUP  (True Parameters)
# -----------------------------------------------------------------------------
np.random.seed(42)

# True physical system parameters
DATA_FILE_NAME = "fixed_controlset.csv"
TRUE_R         = 0.046                   # Intrinsic growth rate
TRUE_K         = 47.81                   # Carrying capacity
TRUE_Y0        = 14.326143333333334      # Initial population size
TRUE_SIGMA     = 0.15                    # Multiplicative log-normal noise scale
ODE            = "gompertz"
TREATMENT      = 4

# -----------------------------------------------------------------------------
# PINN / correction-network config
# -----------------------------------------------------------------------------
# Enfoque híbrido: r y K siguen siendo los parámetros mecanísticos "interpretables".
# La red pequeña solo aprende una corrección residual multiplicativa (en escala log)
# sobre la curva mecanística: y_pred(t) = y_mecanistico(t) * exp(NN(t)).
# NN(t) = 0 => sin corrección (el modelo se reduce al original si USE_NN_CORRECTION=False).
USE_NN_CORRECTION    = True    # apaga esto para recuperar exactamente el modelo original
NN_HIDDEN_UNITS       = 8       # red chica a propósito: es una corrección, no un sustituto de la ODE
NN_WEIGHT_PRIOR_SIGMA = 0.3     # prior angosto centrado en 0 -> actúa como "weight decay" bayesiano,
                                 # evita que la NN domine sobre la parte mecanística

# ---- Physics-informed residual (lo que hace que esto sea un PINN y no solo un "corrector") ----
# y_mecanistico(t) ya es la solucion analitica exacta de la ODE, asi que su derivada ES
# el lado derecho de la ecuacion (RHS). Al agregar la correccion NN, la curva total
# y_total(t) = y_mecanistico(t) * exp(NN(t)) deja de satisfacer esa ODE exactamente.
# ENFORCE_PHYSICS_RESIDUAL penaliza (via pm.Potential) que tanto se desvia, para que la
# NN solo aporte lo que los datos justifiquen sin romper la ley de crecimiento subyacente.
ENFORCE_PHYSICS_RESIDUAL = True
PHYSICS_RESIDUAL_WEIGHT  = 10.0   # sube esto para forzar mas apego a la fisica, baja para dar mas libertad a la NN
# Subsampling razonable si hay demasiadas muestras (ajusta según tu paciencia/CPU)
MAX_TRAJECTORIES = 3000

def verhulst_rhs_pytensor(y, r, K):
    """Lado derecho de la ODE de Verhulst: dy/dt = r*y*(1 - y/K)."""
    return r * y * (1 - y / K)


def gompertz_rhs_pytensor(y, r, K, eps=1e-12):
    """Lado derecho de la ODE de Gompertz: dy/dt = r*y*ln(K/y)."""
    y_safe = pt.clip(y, eps, np.inf)
    return r * y_safe * pt.log(K / y_safe)


ode_rhs_config = {"verhulst": verhulst_rhs_pytensor, "gompertz": gompertz_rhs_pytensor}

treatments = ["Testigo (T1) Kéfir sin ultrasonicar",
              "15 seg. 20 W/cm2 (T2)",
              "1 min. 20 W/cm2 (T3)",
              "15 seg. 34 W/cm2 (T4)",
              "1 min. 34 W/cm2 (T5)"]
save_path = Path("figures")
config_dict  = {"gompertz": (gompertz_eq_solution, gompertz_latent_eq),
                 "verhulst": (verhulst_eq_solution, verhulst_latent_eq)}
ode_solution, latent_eq_solution = config_dict[ODE]

parameter_names = ['r', 'K', 'sigma']
nn_parameter_names = ['nn_W1', 'nn_b1', 'nn_W2', 'nn_b2'] if USE_NN_CORRECTION else []

true_parameter_values = {
    'r': TRUE_R,
    'K': TRUE_K,
    'Y0': TRUE_Y0,
    'sigma': TRUE_SIGMA,
}

initial_prior_conditions = {
    'r': 0.1,
    'K': 50,
}
prior_sigmas = {
    'r': 0.5,
    'K': 0.3,
    'sigma': 0.3,
}

def build_nn_correction(t_scaled, W1, b1, W2, b2):
    """
    MLP pequeño 1 -> H -> 1 con activación tanh, construido con pytensor para
    que NUTS pueda diferenciar a través de él junto con r, K, sigma.

    Devuelve un vector (n,) que representa una corrección en escala LOG:
        y_corregida(t) = y_mecanistica(t) * exp(nn_correction(t))
    """
    t_col = t_scaled[:, None]                    # (n, 1)
    hidden = pt.tanh(pt.dot(t_col, W1) + b1)      # (n, H)
    out = pt.dot(hidden, W2) + b2                 # (n, 1)
    return out.flatten()                          # (n,)


def build_nn_correction_numpy(t_scaled, W1, b1, W2, b2):
    """Misma red que build_nn_correction pero en NumPy puro, para reconstruir
    trayectorias posteriores fuera del contexto de PyMC (sección 4b)."""
    hidden = np.tanh(t_scaled[:, None] @ W1 + b1)
    out = hidden @ W2 + b2
    return out.flatten()

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
        ax.axvline(posterior_mean, color="black", linewidth=1.5, label="Promedio posterior")
        ax.axvline(hdi_lower, color="tab:orange", linestyle="--", linewidth=1.2, label=f"{int(hdi_prob * 100)}% HDI")
        ax.axvline(hdi_upper, color="tab:orange", linestyle="--", linewidth=1.2)
        if true_value is not None:
            ax.axvline(true_value, color="red", linestyle=":", linewidth=1.8, label="Valor verdadero")

        ax.set_title(f"Distribución posterior de '{parameter_name}'")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="best", fontsize=8)

    for j in range(n_params, len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig.suptitle("Distribuciones posteriores de parámetros", fontsize=14)
    fig.tight_layout()
    return axes


os.makedirs(save_path / ODE / str(TREATMENT), exist_ok=True)

# -----------------------------------------------------------------------------
#  Load data
# -----------------------------------------------------------------------------
treatment = treatments[TREATMENT]
dataset = load_data(DATA_FILE_NAME)
dataset = dataset.drop(columns=["Unnamed: 0"])

df = dataset[dataset["tratamiento"] == treatment]
t0, tf = load_time_domain(dataset)
t_data = np.linspace(t0, tf, len(df))

y_clean = ode_solution(t_data, true_parameter_values)
y_obs = df["biomasa(g/L)"]
FIXED_Y0 = y_obs.iloc[0]

# Normalized time, usado únicamente como input de la red de corrección
# (las NN entrenan mejor con inputs ~[0, 1] que con horas crudas).
t_norm = (t_data - t_data.min()) / (t_data.max() - t_data.min())
DT_NORM_DT = 1.0 / (t_data.max() - t_data.min())   # regla de la cadena: d(t_norm)/d(t)

# IMPORTANTE: usamos una variable "shared" de pytensor (no una constante) porque
# pt.grad necesita un nodo diferenciable de verdad para calcular d(NN)/d(t_norm).
# Validado por separado con diferencia finita (error ~1e-12).
t_norm_tensor = pytensor.shared(t_norm.astype("float64"), name="t_norm_shared")

ode_rhs = ode_rhs_config[ODE]


# -----------------------------------------------------------------------------
# 2. BAYESIAN MODEL SPECIFICATION (híbrido: ODE mecanística + corrección NN)
# -----------------------------------------------------------------------------

with pm.Model() as hybrid_model:
    priors = {}
    for param in ['r', 'K']:            # parámetros mecanísticos que se siguen estimando
        sigma_p = prior_sigmas.get(param)
        initial_condition = initial_prior_conditions.get(param)
        priors[param] = pm.LogNormal(param, mu=np.log(initial_condition), sigma=sigma_p)

    sigma_obs = pm.HalfNormal('sigma', sigma=prior_sigmas.get('sigma'))

    # Y0 sigue siendo una constante fija, no una variable aleatoria
    priors['Y0'] = FIXED_Y0

    # Trayectoria mecanística (idéntica al script original)
    y_mechanistic = latent_eq_solution(t_data, priors)

    if USE_NN_CORRECTION:
        # Pesos bayesianos de la red de corrección. Centrados en 0 con sigma
        # angosto: la red arranca como un "no-op" y solo se aleja de la curva
        # mecanística donde los datos realmente lo pidan.
        W1 = pm.Normal('nn_W1', mu=0.0, sigma=NN_WEIGHT_PRIOR_SIGMA, shape=(1, NN_HIDDEN_UNITS))
        b1 = pm.Normal('nn_b1', mu=0.0, sigma=NN_WEIGHT_PRIOR_SIGMA, shape=(NN_HIDDEN_UNITS,))
        W2 = pm.Normal('nn_W2', mu=0.0, sigma=NN_WEIGHT_PRIOR_SIGMA, shape=(NN_HIDDEN_UNITS, 1))
        b2 = pm.Normal('nn_b2', mu=0.0, sigma=NN_WEIGHT_PRIOR_SIGMA, shape=(1,))

        nn_correction = build_nn_correction(t_norm_tensor, W1, b1, W2, b2)
        pm.Deterministic('nn_correction', nn_correction)   # útil para diagnóstico/plots

        y_latent = y_mechanistic * pt.exp(nn_correction)

        if ENFORCE_PHYSICS_RESIDUAL:
            # --- Residual físico (esto es lo que hace que la corrección sea un PINN) ---
            # d(y_mecanistico)/dt es EXACTA (y_mecanistico resuelve la ODE por construcción):
            dy_mechanistic_dt = ode_rhs(y_mechanistic, priors['r'], priors['K'])

            # d(correccion)/dt via autodiff de pytensor. Usamos el truco de "batched grad":
            # como cada NN(t_i) depende solo de su propio t_i (evaluacion vectorizada,
            # sin mezcla entre filas), d(sum(NN))/d(t_norm) recupera element-wise dNN_i/d(t_norm_i).
            d_correction_d_tnorm = pt.grad(pt.sum(nn_correction), t_norm_tensor)
            d_correction_dt = d_correction_d_tnorm * DT_NORM_DT   # regla de la cadena a horas reales

            # Regla del producto sobre y_total = y_mecanistico * exp(correccion):
            dy_latent_dt = pt.exp(nn_correction) * (dy_mechanistic_dt + y_mechanistic * d_correction_dt)

            # Que tanto se aleja la curva TOTAL (con correccion) de seguir cumpliendo
            # la misma ley de crecimiento (r, K) que ya tiene sus propios priors.
            physics_residual = dy_latent_dt - ode_rhs(y_latent, priors['r'], priors['K'])
            pm.Deterministic('physics_residual', physics_residual)

            # Log-densidad extra: -0.5 * peso * sum(residual^2). Entre mayor
            # PHYSICS_RESIDUAL_WEIGHT, mas se penaliza que la NN "invente" dinamica
            # que no es consistente con la ODE mecanistica.
            pm.Potential(
                'physics_loss',
                -0.5 * PHYSICS_RESIDUAL_WEIGHT * pt.sum(pt.sqr(physics_residual)),
            )
    else:
        y_latent = y_mechanistic

    pm.Deterministic('y_latent', y_latent)

    # Verosimilitud explícita sobre la trayectoria (mecanística + corrección).
    # NOTA: la hago explícita aquí en vez de asumir que latent_eq_solution ya la
    # crea internamente, porque esa función solo recibe (t_data, priors) y no
    # tiene acceso a `sigma_obs` ni a `y_obs` de este script. Si tu versión de
    # latent_eq_solution SÍ registra su propia variable observada, elimina una
    # de las dos para no duplicar la verosimilitud.
    pm.LogNormal('y_obs_likelihood', mu=pt.log(y_latent), sigma=sigma_obs, observed=y_obs.values)

    # -------------------------------------------------------------------------
    # 3. RUN MCMC SAMPLING
    # -------------------------------------------------------------------------
    print("Running MCMC Sampler...")
    # cores=os.cpu_count() explícito: en algunos contenedores/entornos con límites de
    # CPU no estándar, pm.sample() falla al autodetectar núcleos (ZeroDivisionError).
    trace = pm.sample(draws=10000, tune=1000, target_accept=0.95,
                       cores=max(1, os.cpu_count() or 1), return_inferencedata=True)


# -----------------------------------------------------------------------------
# 4. RESULTS ANALYSIS
# -----------------------------------------------------------------------------

# Print text summary of parameter posteriors and convergence metrics (R-hat)
print("\n--- Posterior Parameter Summary (parámetros mecanísticos) ---")

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

if USE_NN_CORRECTION:
    # Resumen aparte para los pesos de la NN: no son individualmente
    # interpretables (son parámetros de una red), así que aquí solo nos
    # interesa el diagnóstico de convergencia (r_hat, ess), no su "valor verdadero".
    print("\n--- Posterior Summary (pesos de la red de corrección) ---")
    try:
        nn_summary = az.summary(trace, var_names=nn_parameter_names, hdi_prob=0.94)
    except TypeError:
        nn_summary = az.summary(trace, var_names=nn_parameter_names, ci_prob=0.94, ci_kind='hdi')
    nn_columns = [c for c in ['mean', 'sd', 'r_hat', 'ess_bulk'] if c in nn_summary.columns]
    print(nn_summary[nn_columns])

# Posterior distributions for each mechanistic parameter
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
            ax.set_title(f"Distribución posterior de '{parameter_name}'")
    plt.suptitle("Distribuciones posteriores por parámetro", fontsize=14)
    plt.tight_layout()
else:
    plot_posterior_fallback(
        trace,
        var_names=parameter_names,
        true_values=true_parameter_values,
        hdi_prob=0.94,
        figsize=(12, 8),
    )
plt.savefig(save_path / ODE / str(TREATMENT) / "posterior_distributions.png")

# -----------------------------------------------------------------------------
# 4b. CREDIBLE BANDS (latente + predictiva), incluyendo la corrección NN
# -----------------------------------------------------------------------------

posterior = trace.posterior
samples_dict = {}
for param in ['r', 'K', 'sigma']:
    if param in posterior:
        samples_dict[param] = posterior[param].values.flatten()

r_samples = samples_dict.get('r')
sigma_samples = samples_dict.get('sigma')
n_samples_total = len(r_samples)

# Pesos de la NN por muestra posterior (si aplica). Se aplanan las dos
# dimensiones de cadena/draw en una sola, manteniendo la forma de cada peso.
nn_samples = {}
if USE_NN_CORRECTION:
    for name in nn_parameter_names:
        raw = posterior[name].values                 # (chain, draw, *param_shape)
        nn_samples[name] = raw.reshape(-1, *raw.shape[2:])


if n_samples_total > MAX_TRAJECTORIES:
    idx = np.random.choice(n_samples_total, size=MAX_TRAJECTORIES, replace=False)
else:
    idx = np.arange(n_samples_total)

t_plot = np.linspace(0, t_data.max(), 300)
t_plot_norm = (t_plot - t_data.min()) / (t_data.max() - t_data.min())

latent_trajectories = np.zeros((len(idx), len(t_plot)))
predictive_trajectories = np.zeros((len(idx), len(t_plot)))

for row, i in enumerate(idx):
    parameters = {param: samples_dict[param][i] for param in ['r', 'K']}
    parameters['Y0'] = FIXED_Y0
    mean_traj = ode_solution(t_plot, parameters)

    if USE_NN_CORRECTION:
        correction = build_nn_correction_numpy(
            t_plot_norm,
            nn_samples['nn_W1'][i],
            nn_samples['nn_b1'][i],
            nn_samples['nn_W2'][i],
            nn_samples['nn_b2'][i],
        )
        mean_traj = mean_traj * np.exp(correction)

    latent_trajectories[row, :] = mean_traj

    sigma_i = sigma_samples[i]
    predictive_trajectories[row, :] = mean_traj * np.random.lognormal(
        mean=0.0, sigma=sigma_i, size=len(t_plot)
    )

HDI_PROB = 0.94
lower_q = (1 - HDI_PROB) / 2 * 100   # e.g. 3
upper_q = 100 - lower_q              # e.g. 97

# Banda latente (incertidumbre en los parámetros + pesos de la NN)
latent_lower = np.percentile(latent_trajectories, lower_q, axis=0)
latent_upper = np.percentile(latent_trajectories, upper_q, axis=0)
latent_median = np.percentile(latent_trajectories, 50, axis=0)

# Banda predictiva (incertidumbre en parámetros + ruido de observación)
pred_lower = np.percentile(predictive_trajectories, lower_q, axis=0)
pred_upper = np.percentile(predictive_trajectories, upper_q, axis=0)

# -----------------------------------------------------------------------------
# 4c. PLOT: Data + credible bands + true curve
# -----------------------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.fill_between(t_plot, pred_lower, pred_upper,
                  color='gray', alpha=0.25, zorder=1,
                  label=f'{int(HDI_PROB*100)}% Banda predictiva')

plt.fill_between(t_plot, latent_lower, latent_upper,
                  color='cyan', alpha=0.4, zorder=2,
                  label=f'{int(HDI_PROB*100)}% Banda de credibilidad (latente)')

plt.plot(t_plot, latent_median, color='blue', linewidth=1.5, zorder=3,
         label='Mediana Posterior')

plt.scatter(t_data, y_obs, color='black', zorder=5, s=20, label='Datos Observados')

plt.plot(t_plot, ode_solution(t_plot, true_parameter_values),
         color='red', linestyle='--', linewidth=2, zorder=4, label='Curva estimada (mecanística pura)')

modelo_str = f"{ODE.capitalize()} + corrección NN" if USE_NN_CORRECTION else ODE.capitalize()
plt.title(f"Estimación bayesiana MCMC ({modelo_str})", fontsize=14)
plt.xlabel("Tiempo (horas)", fontsize=12)
plt.ylabel("Biomasa de kéfir de agua (g/L)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.savefig(save_path / ODE / str(TREATMENT) / "plot.png")

# -----------------------------------------------------------------------------
# 4d. PLOT: solo la corrección aprendida por la NN (útil para ver qué "arregla")
# -----------------------------------------------------------------------------
if USE_NN_CORRECTION:
    correction_median = np.zeros(len(t_plot))
    correction_samples = np.zeros((len(idx), len(t_plot)))
    for row, i in enumerate(idx):
        correction_samples[row, :] = build_nn_correction_numpy(
            t_plot_norm,
            nn_samples['nn_W1'][i],
            nn_samples['nn_b1'][i],
            nn_samples['nn_W2'][i],
            nn_samples['nn_b2'][i],
        )
    correction_lower = np.percentile(correction_samples, lower_q, axis=0)
    correction_upper = np.percentile(correction_samples, upper_q, axis=0)
    correction_median = np.percentile(correction_samples, 50, axis=0)

    plt.figure(figsize=(10, 4))
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.fill_between(t_plot, correction_lower, correction_upper, color='purple', alpha=0.3,
                      label=f'{int(HDI_PROB*100)}% HDI corrección')
    plt.plot(t_plot, correction_median, color='purple', linewidth=1.5, label='Mediana corrección (log-escala)')
    plt.title("Corrección residual aprendida por la red (escala log)", fontsize=13)
    plt.xlabel("Tiempo (horas)")
    plt.ylabel("log(corrección)")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / ODE / str(TREATMENT) / "nn_correction.png")

# -----------------------------------------------------------------------------
# 5. MÉTRICAS DE BONDAD DE AJUSTE (RMSE, MAE, MAPE, R^2, AIC, BIC)
# -----------------------------------------------------------------------------
# Todas las métricas se calculan comparando y_obs contra la curva ajustada
# EVALUADA en los mismos tiempos observados (t_data), usando como "punto
# estimado" la media posterior de cada parámetro (r, K, sigma, y pesos NN
# si aplica). Esto es la convención habitual para reportar AIC/BIC de un
# modelo que se ajustó por MCMC: se hace un "plug-in" en la media posterior
# en vez de usar el óptimo de máxima verosimilitud clásico.
print("\n--- Métricas de bondad de ajuste (sobre los datos observados) ---")

r_mean = float(np.mean(samples_dict['r']))
K_mean = float(np.mean(samples_dict['K']))
sigma_mean = float(np.mean(sigma_samples))

point_parameters = {'r': r_mean, 'K': K_mean, 'Y0': FIXED_Y0}
y_pred_mechanistic = ode_solution(t_data, point_parameters)

if USE_NN_CORRECTION:
    nn_mean_weights = {name: np.mean(nn_samples[name], axis=0) for name in nn_parameter_names}
    correction_at_data = build_nn_correction_numpy(
        t_norm,
        nn_mean_weights['nn_W1'],
        nn_mean_weights['nn_b1'],
        nn_mean_weights['nn_W2'],
        nn_mean_weights['nn_b2'],
    )
    y_pred = y_pred_mechanistic * np.exp(correction_at_data)
    # k efectivo de la NN: W1 (1*H) + b1 (H) + W2 (H*1) + b2 (1) = 3H + 1
    n_nn_params = sum(w.size for w in nn_mean_weights.values())
else:
    y_pred = y_pred_mechanistic
    n_nn_params = 0

y_obs_values = y_obs.values.astype(float)
residuals = y_obs_values - y_pred

n_obs = len(y_obs_values)
# Parámetros "libres" del modelo: r, K, sigma_obs, + pesos de la NN si aplica.
n_params = 3 + n_nn_params

rmse = float(np.sqrt(np.mean(residuals ** 2)))
mae = float(np.mean(np.abs(residuals)))
# MAPE en %. Válido aquí porque la biomasa observada es siempre > 0.
mape = float(np.mean(np.abs(residuals / y_obs_values)) * 100)

ss_res = float(np.sum(residuals ** 2))
ss_tot = float(np.sum((y_obs_values - np.mean(y_obs_values)) ** 2))
r_squared = 1.0 - ss_res / ss_tot

# Log-verosimilitud evaluada en la media posterior, bajo la MISMA
# verosimilitud LogNormal(mu=log(y_pred), sigma=sigma_mean) del modelo:
#   log p(y | mu, sigma) = -log(y * sigma * sqrt(2*pi)) - (log(y) - mu)^2 / (2*sigma^2)
log_lik_per_point = (
    -np.log(y_obs_values * sigma_mean * np.sqrt(2 * np.pi))
    - (np.log(y_obs_values) - np.log(y_pred)) ** 2 / (2 * sigma_mean ** 2)
)
log_lik_total = float(np.sum(log_lik_per_point))

aic = 2 * n_params - 2 * log_lik_total
bic = n_params * np.log(n_obs) - 2 * log_lik_total

metrics_summary = {
    'RMSE': rmse,
    'MAE': mae,
    'MAPE (%)': mape,
    'R^2': r_squared,
    'Log-verosimilitud (media posterior)': log_lik_total,
    'AIC': aic,
    'BIC': bic,
    'n_observaciones': n_obs,
    'n_parametros': n_params,
}

for metric_name, metric_value in metrics_summary.items():
    if isinstance(metric_value, (int, np.integer)):
        print(f"{metric_name:40s}: {metric_value}")
    else:
        print(f"{metric_name:40s}: {metric_value:.4f}")

metrics_path = save_path / ODE / str(TREATMENT) / "fit_metrics.txt"
with open(metrics_path, "w") as f:
    for metric_name, metric_value in metrics_summary.items():
        if isinstance(metric_value, (int, np.integer)):
            f.write(f"{metric_name}: {metric_value}\n")
        else:
            f.write(f"{metric_name}: {metric_value:.6f}\n")
print(f"\nMétricas guardadas en: {metrics_path}")

# -----------------------------------------------------------------------------
# 5b. (Opcional) WAIC / LOO: alternativa plenamente bayesiana a AIC/BIC
# -----------------------------------------------------------------------------
# AIC y BIC son métricas "frecuentistas" adaptadas aquí vía plug-in de la
# media posterior; no usan toda la información de la posterior. Si el objetivo
# es comparar modelos de forma más rigurosa dentro de un marco bayesiano,
# WAIC/LOO (que sí usan la distribución posterior completa) son preferibles.
# Se calculan aquí como complemento, sin reemplazar las métricas de arriba.
try:
    with hybrid_model:
        pm.compute_log_likelihood(trace, var_names=['y_obs_likelihood'])
    waic_result = az.waic(trace)
    loo_result = az.loo(trace)
    print("\n--- WAIC / LOO (alternativa bayesiana a AIC/BIC) ---")
    print(waic_result)
    print(loo_result)
    with open(save_path / ODE / str(TREATMENT) / "fit_metrics.txt", "a") as f:
        f.write(f"\nWAIC:\n{waic_result}\n\nLOO:\n{loo_result}\n")
except Exception as exc:
    print(f"\n(No se pudieron calcular WAIC/LOO: {exc})")