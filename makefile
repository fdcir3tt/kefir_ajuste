phys_disc:
	poetry run python scripts/physics_discovery.py

phys_disc_nn:
	poetry run python scripts/physics_discovery_nn.py

stability:
	poetry run python scripts/seed_stability.py
ui:
	poetry run mlflow ui

confidence: 
	poetry run python scripts/mcmc.py

b_pinn: 
	poetry run python scripts/bayesian_pinn.py