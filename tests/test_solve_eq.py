from src import runge_kutta
import numpy as np


def test_runge_kutta():
    # Output length
    def f1(x, y): return y

    expected_len=201
    result = runge_kutta(f1, y0=1, interval=(0, 1), n=expected_len)
    assert len(result) == expected_len, f"Expect length {expected_len}, got { len(result) }"

    # Approximate solution to dy/dx = y, y(0) = 1

    result = runge_kutta(f1, y0=1, interval=(0, 1), n=10)
    for x, y in result:
        expected = np.exp(x)
        tolerance = 0.01  # Small error tolerance
        assert abs(y - expected) < tolerance, f"At x={x}, expected ~{expected}, got {y}"



    print("Tests completed successfully!")
