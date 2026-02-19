import matplotlib.pyplot as plt
import numpy as np
import os
from src import runge_kutta

os.makedirs("plots",exist_ok=True)

# Diff Eq. Parameters
R=0.1
K=40

# Runge-Kutta Parameters
domain=( 1, 200 )
n_points=50
Y_0=1

# Logarithmic Differential Equation
def log_dif_eq(y:float,x:float,r:float=0.01,k:float=100)->float:
    """
      Logistic differential equation.

      Computes the derivative dy/dx of the logistic growth model:

          dy/dx = r * y * (1 - y / k)

      Parameters
      ----------
      y : float
          Current population value.
      x : float
          Placeholder for the independent variable (not used in this equation,
          but kept for compatibility with ODE solvers that expect f(y, x)).
      r : float, optional
          Intrinsic growth rate (default is 0.01).
      k : float, optional
          Carrying capacity of the system (default is 100).

      Returns
      -------
      float
          Derivative dy/dx at the given value of y.
      """


    return r*y*(1-y/k)

def analytic_sol(y0:float,interval:tuple,r:float=0.01,k:float=100)->list:
    """
        Analytical solution of the logistic growth equation.

        Generates the solution curve for the logistic growth model:

            y(x) = (y0 * k) / (y0 + (k - y0) * exp(-r * x))

        The curve is sampled over the given interval.

        Parameters
        ----------
        y0 : float
            Initial value of the population at x = 0.
        interval : tuple
            Two-element tuple (x_start, x_end) defining the domain of the solution.
        r : float, optional
            Intrinsic growth rate (default is 0.01).
        k : float, optional
            Carrying capacity of the system (default is 100).

        Returns
        -------
        list of tuple
            A list of (x, y) pairs representing the solution curve sampled at
            1000 evenly spaced points across the interval.
        """
    x_values = np.linspace(interval[0], interval[1], 1000)
    curve = [  ( x, y0* k / (y0 + (k - y0) * np.exp(- r * x)) ) for x in x_values]
    return curve

f=lambda x, y: log_dif_eq(y=y, x=x, r=R, k=K)



numeric_solution = runge_kutta(f=f,y0=Y_0, interval=domain,n=n_points)
analytic_solution = analytic_sol(y0=Y_0, interval=domain,r=R,k=K)



# Plot

x_numeric, y_numeric = zip(*numeric_solution)
x_analytic, y_analytic = zip(*analytic_solution)

plt.figure(figsize=(10, 5))
plt.plot(x_numeric, y_numeric, label='Numerical Solution (Runge-Kutta)', color='blue', marker='o')
plt.plot(x_analytic, y_analytic, label='Analytical Solution', color='red', linewidth=2)
plt.title("Logistic Differential Equation Solution")
plt.xlabel("Time (hours)  ")
plt.ylabel("Population density (g/cm^3)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/log_eq.png")