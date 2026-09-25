.PHONY: help install lint format test train pipeline run docker drift smoke

help:  ## list targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /  -  /'

install:  ## install pinned dependencies into the active environment
	pip install -r requirements.txt

lint:  ## ruff + black --check (same as CI)
	ruff check src tests scripts
	black --check src tests scripts

format:  ## auto-format and auto-fix
	black src tests scripts
	ruff check --fix src tests scripts

test:  ## run the test suite
	pytest -q

train:  ## ingest -> features -> train, outside DVC
	python -m src.data.ingest
	python -m src.features.build_features
	python -m src.models.train

pipeline:  ## reproduce via DVC (only changed stages re-run)
	dvc repro

run:  ## serve the API locally with reload
	uvicorn src.api.main:app --reload

docker:  ## build the image locally
	docker build -t temperature-forecast:latest .

drift:  ## write reports/drift.html with Evidently
	python -m src.monitoring.drift_report

smoke:  ## smoke-test a deployment: make smoke URL=https://...
	python scripts/smoke_test.py $(URL)
