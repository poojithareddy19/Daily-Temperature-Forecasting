# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-09-25

First production release: a forecasting microservice with the full MLOps machinery
around it, built in 30 daily steps.

### Added

- Data pipeline: ingestion from a public URL, fail-fast validation, lag, rolling-mean and
  calendar feature engineering, a leak-free temporal train/test split, RMSE and MAE with a
  persistence baseline.
- RandomForest model trained from `params.yaml`, persisted together with its feature list so
  serving can never mismatch training.
- DVC: raw data versioned, `prepare` and `train` stages in `dvc.yaml`, metrics tracked in
  `metrics.json`, everything reproducible with `dvc repro`.
- MLflow experiment tracking for every training run (params, metrics, model artifact).
- FastAPI service with `/health`, `/predict` (Pydantic request and response contracts, clean
  422 and 503 errors, request logging) and a Prometheus `/metrics` endpoint.
- Tests: unit tests for features, metrics and split; API tests with a stubbed model;
  metrics endpoint test. Ruff, Black and pre-commit hooks enforce style.
- Multi-stage Dockerfile that trains the model during the build, plus docker-compose for a
  local API and MLflow server.
- GitHub Actions: CI (lint and tests on every push and PR) and CD (build the image and push
  it to GHCR with `latest` and `sha-<commit>` tags on every merge to `main`).
- Branch ruleset on `main`: pull requests required, `quality` check required, force pushes
  and deletions blocked.
- Evidently data-drift report comparing recent data against a reference window.
- Render blueprint (`render.yaml`) with health-checked auto-deploys, a post-deploy smoke
  test script, and status badges.
- Full README and a Makefile of common commands.

### Fixed

- `pathspec` pinned to 0.12.1 so DVC 3.55 imports on fresh installs.
- `metrics.json` written with a trailing newline so pre-commit and DVC agree on its hash.

[1.0.0]: https://github.com/poojithareddy19/Daily-Temperature-Forecasting/releases/tag/v1.0.0
