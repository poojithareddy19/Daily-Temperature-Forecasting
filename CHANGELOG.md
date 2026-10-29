# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Four naive baselines (persistence, 7-day mean, climatology, same day last year) logged
  next to the model in `metrics.json` and MLflow.
- Ridge regression as a second model, chosen with `train.model` in `params.yaml`.
- Walk-forward backtest over 1986-1990 as a DVC stage, plus RMSE by month and a residual plot.
- Train/serve feature parity test.
- Model sha256 in `metrics.json` and in the `/health` response.
- Prometheus and Grafana in docker compose, with a provisioned dashboard for request rate
  and p95 latency, and `scripts/send_requests.py` to generate traffic.
- `/predict` logs each request's features to `logs/requests.jsonl`; the drift report compares
  them with the training features (falls back to the test period below 100 requests).
- Weekly drift report workflow that uploads `drift.html` as an artifact.
- `ingest` DVC stage: downloads the raw CSV and checks it against `data.raw_sha256` in
  `configs/config.yaml` before writing, so a fresh clone rebuilds with a single `dvc repro`.
- Tests for data validation, the ingest checksum and the `/predict` input rules.
- `.gitattributes` that checks text files out with LF, so DVC hashes match on every OS.

### Changed

- Production model is now Ridge instead of RandomForest: lower mean backtest RMSE
  (2.28 vs 2.33) and a model file of about 1 KB instead of 96 MB.
- Calendar features: dropped `month` and `dayofweek`, added sin/cos day of year.
- `/predict` builds its features with the same functions as training.
- `/health` returns 503 when no model is loaded.
- Startup moved from the deprecated `on_event` hook to a lifespan handler.
- README describes the metrics endpoint and drift report as what they are.
- The raw CSV is a DVC stage output instead of a `.dvc`-tracked file (there was no remote
  to pull it from), and is written with LF line endings.
- `/predict` takes `date` as a strict `YYYY-MM-DD` string and `recent_temps` as 30 to 366
  finite values within -50..60 deg C; the response is unchanged.
- The `api` service in docker compose no longer sets `MLFLOW_TRACKING_URI` or waits for
  MLflow, which it never used; the MLflow service remains as a UI for `./mlruns`.
- Pipeline outputs regenerated with Python 3.11, the CI and Docker interpreter: metrics
  are unchanged except `mae`, which moves in the 16th decimal, and the model hash.

### Fixed

- `src/data/validate.py` now runs in the `prepare` stage and raises `DataValidationError`
  instead of using `assert` (stripped by `python -O`); malformed readings fail the build
  instead of being dropped silently.
- `/predict` no longer returns a forecast for physically impossible input such as thirty
  readings of 999 deg C, and a NaN or infinite reading returns 422 instead of a 500.
- `matplotlib` is pinned in `requirements.txt` (the error analysis stage imports it; it was
  only installed through MLflow).
- The walk-forward backtest trains single-threaded, so its prediction files are identical
  on every run and a fresh `dvc repro` no longer rewrites their hashes.
- `features.csv` is written with LF line endings, so its hash matches on every OS.
- Plain `make train` and runs logged through the compose MLflow server share one
  experiment in either order; artifacts go to `./mlartifacts`.
- The web UI forecasts the day after the newest fetched reading instead of a free date
  picker that could disagree with the history.
- README notes the two missing calendar days and their effect on the lag features.

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
