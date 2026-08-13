import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Growth models
# ---------------------------------------------------------

def verhulst(t, K, r, N0):
    """Verhulst / Logistic model."""
    return K / (1 + ((K - N0) / N0) * np.exp(-r * t))


def gompertz(t, K, r, N0):
    """Gompertz model."""
    return K * np.exp(np.log(N0 / K) * np.exp(-r * t))


def richards(t, K, r, N0, nu):
    """Richards model."""
    A = (K / N0)**nu - 1
    return K / (1 + A * np.exp(-r * nu * t))**(1 / nu)


# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------

K = 500       # Carrying capacity
r = 0.2        # Growth rate
N0 = 10        # Initial population
nu = 0.5      # Richards shape parameter

# Time values
t = np.linspace(0, 50, 500)


# ---------------------------------------------------------
# Calculate model predictions
# ---------------------------------------------------------

N_verhulst = verhulst(t, K, r, N0)
N_gompertz = gompertz(t, K, r, N0)
N_richards = richards(t, K, r, N0, nu)


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

sns.set_theme(style="whitegrid", context="talk")

plt.figure(figsize=(11, 7))

sns.lineplot(
    x=t,
    y=N_verhulst,
    linewidth=2.5,
    label=f"Verhulst  ($K={K}, r={r}, N_0={N0}$)"
)

sns.lineplot(
    x=t,
    y=N_gompertz,
    linewidth=2.5,
    label=f"Gompertz  ($K={K}, r={r}, N_0={N0}$)"
)

sns.lineplot(
    x=t,
    y=N_richards,
    linewidth=2.5,
    label=f"Richards  ($K={K}, r={r},N_0={N0}, \\nu={nu}$)"
)


# ---------------------------------------------------------
# Labels and formatting
# ---------------------------------------------------------

plt.xlabel("Tiempo, t")
plt.ylabel("Población, N(t)")
plt.title("Comparasión de modelos Verhulst, Gompertz, y Richards ")

plt.legend(
    title="Parámetros",
    fontsize=11,
    title_fontsize=12,
    loc="lower right"
)

plt.tight_layout()
plt.savefig("figures/growth_models.png")
plt.show()