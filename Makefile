.PHONY: test toy

test:
	PYTHONPATH=src pytest -q

toy:
	python scripts/make_toy_jsonl.py --out data/train.jsonl --repeat 256
