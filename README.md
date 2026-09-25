# MLOps: Daily Temperature Forecasting

![CI](https://github.com/poojithareddy19/Daily-Temperature-Forecasting/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/poojithareddy19/Daily-Temperature-Forecasting/actions/workflows/cd.yml/badge.svg)

An end-to-end, production-grade MLOps project built from scratch over 30 days. Forecasts the next day's minimum temperature and ships as a containerized API.

## Development

Install the git hooks once so ruff, black and basic file checks run on every commit:

```bash
pip install -r requirements.txt
pre-commit install
pre-commit run --all-files   # first run checks the whole repo
```

### Branch protection

`main` is the deployable branch. On GitHub, open Settings > Branches > Add branch ruleset for `main` and enable:

- Require a pull request before merging
- Require status checks to pass, selecting the `CI / quality` check
- Block force pushes

With this rule in place nothing reaches `main` without a green CI run.
