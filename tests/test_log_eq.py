from src import log_dif_eq,analytic_sol
import numpy as np

def test_log_dif_eq_growth():
    """Test that derivative is positive when y < k and negative when y > k."""
    r, k = 0.1, 50
    y_small = 10
    y_large = 100

    # when y < k, growth should be positive
    assert log_dif_eq(y_small, None, r=r, k=k) > 0

    # when y > k, growth should be negative
    assert log_dif_eq(y_large, None, r=r, k=k) < 0


def test_log_dif_eq_equilibrium():
    """Test equilibrium points at y=0 and y=k."""
    r, k = 0.1, 50
    assert np.isclose(log_dif_eq(0, None, r=r, k=k), 0.0)
    assert np.isclose(log_dif_eq(k, None, r=r, k=k), 0.0)


def test_analytic_sol_initial_condition():
    """At x=0, solution should equal y0."""
    y0, r, k = 5, 0.1, 100
    curve = analytic_sol(y0, (0, 10), r=r, k=k)
    x0, y0_computed = curve[0]
    assert np.isclose(x0, 0.0)
    assert np.isclose(y0_computed, y0)


def test_analytic_sol_asymptote():
    """As x grows large, solution should approach k."""

    y0, r, k = 5, 0.1, 100
    curve = analytic_sol(y0, (0, 100), r=r, k=k)
    _, y_end = curve[-1]
    assert np.isclose(y_end, k, atol=1e-1)


def test_analytic_sol_monotonic_increase():
    """Solution should be increasing if 0 < y0 < k."""
    y0, r, k = 5, 0.1, 100
    curve = analytic_sol(y0, (0, 10), r=r, k=k)
    y_values = [y for _, y in curve]
    assert all(y_values[i] <= y_values[i+1] for i in range(len(y_values)-1))


