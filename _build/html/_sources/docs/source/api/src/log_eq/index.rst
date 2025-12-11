src.log_eq
==========

.. py:module:: src.log_eq






Module Contents
---------------

.. py:data:: R
   :value: 0.1


.. py:data:: K
   :value: 40


.. py:data:: domain
   :value: (1, 200)


.. py:data:: n_points
   :value: 50


.. py:data:: Y_0
   :value: 1


.. py:function:: log_dif_eq(y: float, x: float, r: float = 0.01, k: float = 100) -> float

   Logistic differential equation.

   Computes the derivative dy/dx of the logistic growth model:

       dy/dx = r * y * (1 - y / k)

   :param y: Current population value.
   :type y: float
   :param x: Placeholder for the independent variable (not used in this equation,
             but kept for compatibility with ODE solvers that expect f(y, x)).
   :type x: float
   :param r: Intrinsic growth rate (default is 0.01).
   :type r: float, optional
   :param k: Carrying capacity of the system (default is 100).
   :type k: float, optional

   :returns: Derivative dy/dx at the given value of y.
   :rtype: float


.. py:function:: analytic_sol(y0: float, interval: tuple, r: float = 0.01, k: float = 100) -> list

   Analytical solution of the logistic growth equation.

   Generates the solution curve for the logistic growth model:

       y(x) = (y0 * k) / (y0 + (k - y0) * exp(-r * x))

   The curve is sampled over the given interval.

   :param y0: Initial value of the population at x = 0.
   :type y0: float
   :param interval: Two-element tuple (x_start, x_end) defining the domain of the solution.
   :type interval: tuple
   :param r: Intrinsic growth rate (default is 0.01).
   :type r: float, optional
   :param k: Carrying capacity of the system (default is 100).
   :type k: float, optional

   :returns: A list of (x, y) pairs representing the solution curve sampled at
             1000 evenly spaced points across the interval.
   :rtype: list of tuple


.. py:data:: f

.. py:data:: numeric_solution
   :value: []


.. py:data:: analytic_solution

