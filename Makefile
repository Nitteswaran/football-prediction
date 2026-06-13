PY := .venv/bin/python

.PHONY: setup data features tune train evaluate simulate api test all refresh

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

data:
	curl -sL -o data/raw/results.csv https://raw.githubusercontent.com/martj42/international_results/master/results.csv
	curl -sL -o data/raw/shootouts.csv https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv
	curl -sL -o data/raw/goalscorers.csv https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv

features:
	$(PY) -m features

tune:
	$(PY) -m training.tune --trials 100

train:
	$(PY) -m training.train

evaluate:
	$(PY) -m evaluation.report

simulate:
	$(PY) -m simulation.engine --sims 100000

api:
	.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000

# production: run behind a TLS-terminating reverse proxy (nginx/Caddy/platform)
serve:
	.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 \
		--workers 2 --proxy-headers --forwarded-allow-ips="*" --no-server-header

test:
	$(PY) -m pytest tests/ -q

all: features train evaluate simulate

# re-pull results, retrain, and ship to the Space only if accuracy holds
refresh:
	bash scripts/refresh.sh
