# Commands used in the README. Windows users without make: run the command after each target.
PY ?= python
IMAGE ?= corvita-icu:v1

.PHONY: data day1 train baseline test api docker-build docker-run monitor lock

data:  ## download sources, verify against PhysioNet SHA256SUMS, write/verify data/manifest.json
	$(PY) -m src.ingest

day1: data  ## parse, validate, 24 h features, splits
	$(PY) -m src.run_day1

train:  ## choose on val, score test, save models/v1 (needs a clean git tree)
	$(PY) -m src.train

baseline:  ## monitoring reference for the current model version
	$(PY) -m src.monitor --make-baseline

test:
	$(PY) -m pytest -q

api:  ## local API with request logs in logs/requests.jsonl
	LOG_PATH=logs/requests.jsonl $(PY) -m uvicorn api.main:create_app --factory --port 8000 --no-access-log

docker-build:
	docker build -t $(IMAGE) .

docker-run:  ## request logs land in ./logs on the host
	docker run --rm -p 8000:8000 -e LOG_PATH=/app/logs/requests.jsonl -v "$(CURDIR)/logs:/app/logs" $(IMAGE)

monitor:  ## exit code 0 = ok or not enough traffic, 1 = alert, 2 = no log file
	$(PY) -m src.monitor --log logs/requests.jsonl

lock:  ## after changing requirements*.txt, from the environment used for training
	$(PY) -m pip freeze --exclude-editable > requirements.lock
