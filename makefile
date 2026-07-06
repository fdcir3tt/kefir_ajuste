phys_disc:
	poetry run python scripts/physics_discovery.py

ui:
	poetry run mlflow ui

confidence: 
	poetry run python scripts/mcmc.py