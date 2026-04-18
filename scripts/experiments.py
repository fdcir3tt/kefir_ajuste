import subprocess
import sys
from pathlib import Path
import itertools

SCRIPT_PATH = Path("scripts")

treatments = range(1, 5)
grades = range(3, 7)
n_iterations = [8000, 10000, 12000, 15000, 20000]

grid_space = itertools.product(treatments, grades, n_iterations)

processes = []

for treatment, grade, iterations in grid_space:
    # log_polynomial.py
    if treatment > 1:
        subprocess.run([
            sys.executable,
            str(SCRIPT_PATH / "log_polynomial.py"),
            "--treatment", str(treatment),
            "--grade", str(grade),
            "--epochs", str(iterations)
        ])
    
    
    # log.py
    subprocess.run([
        sys.executable,
        str(SCRIPT_PATH / "log.py"),
        "--treatment", str(treatment),
        "--epochs", str(iterations)
    ])
    
