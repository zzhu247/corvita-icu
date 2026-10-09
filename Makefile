PY ?= python

.PHONY: data day1 test

data:	## download source files, write or verify checksums
	$(PY) -m src.ingest

day1: data	## checksums -> parse -> validate -> features -> splits
	$(PY) -m src.run_day1

test:
	$(PY) -m pytest -q

train:	## Day 2: select on val, score test once, save models/v1
  	$(PY) -m src.train
	