.PHONY: install dev data run eval ablate serve test lint clean demo

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

data:
	python scripts/generate_synthetic_scenario.py --out data/scenarios --labels eval/datasets/ground_truth.jsonl --n 24

run: data
	python -m awvi.cli run --scenarios data/scenarios --out data/events.json

eval: run
	python eval/run_eval.py --scenarios data/scenarios --truth eval/datasets/ground_truth.jsonl --events data/events.json

ablate: data
	python eval/ablations.py --scenarios data/scenarios --truth eval/datasets/ground_truth.jsonl

serve: run
	python -m awvi.cli serve --port 8000

demo: eval serve

test:
	pytest

lint:
	ruff check src tests eval scripts

clean:
	rm -rf data/clips data/events.json eval/results.json eval/ablations.json .pytest_cache .ruff_cache
