# MLOps: Daily Temperature Forecasting

![CI](https://github.com/poojithareddy19/Daily-Temperature-Forecasting/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/poojithareddy19/Daily-Temperature-Forecasting/actions/workflows/cd.yml/badge.svg)

End-to-end MLOps around a deliberately simple model: versioned data, tracked experiments,
a reproducible pipeline, a tested and containerized FastAPI service, CI/CD, monitoring,
and a live cloud deployment. The service forecasts the next day's minimum temperature
for Melbourne from the last 30 days of readings.

## Live demo

- Web UI: https://temperature-forecast-api-mejw.onrender.com (fetches recent readings for any place, or paste your own)
- Interactive docs: https://temperature-forecast-api-mejw.onrender.com/docs
- Health: https://temperature-forecast-api-mejw.onrender.com/health

The service runs on Render's free tier, so the first request after a period of idleness
takes up to a minute while the instance wakes up.

```bash
curl -X POST https://temperature-forecast-api-mejw.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "1991-01-01", "recent_temps": [12.0, 11.5, 13.2, 12.8, 14.0, 13.1, 12.2, 11.9, 12.5, 13.0, 12.7, 11.8, 12.1, 13.4, 12.9, 12.3, 11.7, 12.6, 13.3, 12.0, 11.6, 12.4, 13.5, 12.8, 12.2, 11.9, 12.7, 13.1, 12.5, 12.0]}'
```

`recent_temps` holds the last 30 daily minimums, newest first (index 0 is yesterday).

## Architecture

```
public CSV (GitHub raw URL)
        |  python -m src.data.ingest
        v
data/raw  (tracked by DVC, not Git)
        |  dvc repro: prepare -> train
        v
data/processed/features.csv  ->  RandomForest  ->  models/model.pkl + metrics.json
        |                         (params.yaml, every run logged to MLflow)
        v
FastAPI service  /health  /predict  /metrics      <- Docker image, model baked in at build
        |
        |  push to main
        v
GitHub Actions  CI (ruff, black, pytest)  ->  CD (build image, push to GHCR)
        |
        v
Render  (render.yaml blueprint, auto-deploys main, health-checked rollout)
```

CI and CD are separate workflows on purpose: CI protects `main` (nothing merges without a
green check), CD ships whatever lands on `main`.

## Tech stack

| Role | Tool | Why |
|---|---|---|
| Language | Python 3.11 | Universal in ML, fast, supported by every tool below |
| Modeling | pandas, scikit-learn | CPU-only and reliable; the pipeline is the point, not the model |
| Data versioning and pipeline | DVC | Git-like data tracking and a pipeline DAG in one tool |
| Experiment tracking | MLflow | Params, metrics, and model artifacts for every run |
| Serving | FastAPI, Pydantic, uvicorn | Typed, validated contracts with auto-generated docs |
| Quality | pytest, Ruff, Black, pre-commit | Fast tests, fast lint, no formatting debates |
| Packaging | Docker (multi-stage), docker-compose | Identical runtime on laptop, CI, and cloud |
| CI/CD | GitHub Actions, GHCR | Native to the repo; images tagged `latest` and `sha-<commit>` |
| Monitoring | prometheus-client, Evidently | Request metrics and data-drift reports |
| Hosting | Render | Docker deploys straight from GitHub, infrastructure declared in `render.yaml` |

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

make install    # pip install -r requirements.txt
make train      # ingest -> features -> train (writes models/model.pkl and metrics.json)
make test       # pytest
make run        # serve the API on http://127.0.0.1:8000 (docs at /docs)
```

Every target is a thin alias; run `make help` to list them, or read the [Makefile](Makefile).

## Reproducible pipeline

```bash
dvc repro           # re-runs only the stages whose inputs changed
dvc metrics show    # rmse and mae from metrics.json
dvc metrics diff    # compare against the last commit
```

Tuning happens in [params.yaml](params.yaml) only. Because it is a declared dependency of
both stages, editing it and running `dvc repro` retrains and rewrites `dvc.lock`, so any
commit can be rebuilt exactly. Run DVC with the virtual environment activated: its stage
commands call `python`.

Runs are logged to a local MLflow file store; open it with `mlflow ui` from the repo root.

## Results

Evaluated on the most recent 20% of the series (724 days), never shuffled.

| Model | RMSE (deg C) | MAE (deg C) |
|---|---|---|
| Persistence baseline (tomorrow = today) | 2.48 | 1.95 |
| RandomForest, 400 trees, depth 20 | **2.15** | **1.70** |

The baseline is the bar to beat; a model that cannot beat "predict yesterday" is not
earning its complexity.

## Monitoring

- `GET /metrics` exposes Prometheus metrics: `predictions_total` and a
  `prediction_latency_seconds` histogram.
- `python -m src.monitoring.drift_report` compares the most recent 30% of the feature
  table against the first 70% with Evidently and writes `reports/drift.html`.
- Every prediction request is logged with its date and result.

## Project structure

```
.github/workflows/   ci.yml (lint + test), cd.yml (build image, push to GHCR)
configs/config.yaml  paths, data URL, log level
data/                raw and processed data (DVC-tracked, git-ignored)
models/              trained model.pkl (git-ignored, rebuilt by the pipeline)
scripts/             smoke_test.py: hits a deployed service end to end
src/
  config.py          config loader, PROJECT_ROOT
  logger.py          shared structured logger
  data/              ingest.py (download), validate.py (fail-fast checks)
  features/          lags, rolling means, calendar features
  models/            split.py, evaluate.py, train.py
  api/               schemas.py, main.py (/health /predict /metrics)
  monitoring/        drift_report.py
tests/               unit tests for features, metrics, split; API tests with a stubbed model
Dockerfile           multi-stage image; the model is trained during the build
docker-compose.yml   API plus an MLflow server for local use
dvc.yaml / dvc.lock  pipeline stages and pinned input/output hashes
params.yaml          hyperparameters and feature settings
render.yaml          Render blueprint (infrastructure as code)
Makefile             command shortcuts
```

## Deployment

- Merging to `main` triggers CD, which builds the image and pushes it to
  `ghcr.io/poojithareddy19/daily-temperature-forecasting` with `latest` and `sha-<commit>` tags.
- Render watches `main` too and rebuilds the Dockerfile on every merge, then swaps
  traffic only after `/health` passes.
- After a deploy, verify it from the outside:

```bash
python scripts/smoke_test.py https://temperature-forecast-api-mejw.onrender.com
```

## Development

Install the git hooks once so ruff, black and basic file checks run on every commit:

```bash
pre-commit install
pre-commit run --all-files   # first run checks the whole repo
```

`main` is protected by a ruleset: direct pushes are rejected, changes arrive through a pull
request, and the `quality` CI check must pass before merging.

## Roadmap

Scoped out of v1 and documented as next steps: scheduled retraining triggered by drift,
a remote MLflow server with the Model Registry as the deploy gate, Kubernetes with a
HorizontalPodAutoscaler, Terraform for the cloud infrastructure, and Grafana dashboards
on top of the Prometheus metrics. See the changelog for what shipped.

## License

Apache License 2.0. See [LICENSE](LICENSE).
