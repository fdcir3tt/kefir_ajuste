src.solve_eq
============

.. py:module:: src.solve_eq




Module Contents
---------------

.. py:function:: runge_kutta(f, y0: float, interval: tuple, n: int = 10) -> list

   Numerically solves an ordinary differential equation (ODE) using the classical
   4th-order Runge-Kutta method (RK4).

   :param f: The derivative function of the ODE, with signature f(x, y).
             It must accept two arguments:
                 x : float
                     The independent variable.
                 y : float
                     The dependent variable (current estimate of y at x).
   :type f: callable
   :param y0: Initial value of y at the starting point of the interval (y(x0)).
   :type y0: float
   :param interval:
                    A tuple (x0, xn) representing the domain over which to solve the ODE:
                        x0 : float
                            Start of the interval.
                        xn : float
                            End of the interval.
   :type interval: tuple of float
   :param n: Number of steps (subintervals) to divide the interval [x0, xn] into.
             A higher value increases accuracy but requires more computation.
   :type n: int, optional (default=10)

   :returns: A list of (x, y) tuples, where each tuple represents the estimated solution y at a corresponding x.
             The list has (n) points starting from x0 to xn.
   :rtype: list of tuple

   .. rubric:: Notes

   This implementation uses the classical Runge-Kutta 4th-order method:
       k1 = h * f(x, y)
       k2 = h * f(x + h/2, y + k1/2)
       k3 = h * f(x + h/2, y + k2/2)
       k4 = h * f(x + h, y + k3)
       y_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4) / 6

   .. rubric:: Example

   >>> f = lambda x, y: 0.1 * y * (1 - y / 40)
   >>> runge_kutta(f, y0=1, interval=(1, 200), n=50)

   [(1.0, 1.0), (5.0, 1.452), ..., (200.0, 39.99)]


