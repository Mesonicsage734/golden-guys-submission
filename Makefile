SHELL := /bin/bash

.PHONY: setup play arena zip gate commit-gate

setup:
	uv sync
	cp scripts/pre-commit .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

zip:
	uv run python -m harness.package --include book --include syzygy

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000

commit-gate:
	uv run python scripts/commit_gate.py $(if $(GAMES),--games $(GAMES)) $(if $(MIN_SCORE),--min-score $(MIN_SCORE))
