# Common tasks. Run `make help` for the list.
SHELL := /bin/bash
BACKEND := cd backend &&
CLI := $(BACKEND) uv run rag-arena

.PHONY: help hooks install db-up db-down test lint fmt typecheck

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

hooks: ## Enable the repo's git hooks (secret scanning on commit)
	git config core.hooksPath .githooks

install: ## Install backend and web dependencies
	$(BACKEND) uv sync
	cd web && npm ci

db-up: ## Start Postgres + pgvector on port 5433
	docker compose up -d --wait db

db-down: ## Stop the project's containers
	docker compose down

test: ## Run the offline backend tests
	$(BACKEND) uv run pytest

lint: ## Lint backend and web
	$(BACKEND) uv run ruff check . && uv run ruff format --check .
	cd web && npm run lint && npm run format:check

fmt: ## Format backend and web
	$(BACKEND) uv run ruff format . && uv run ruff check --fix .
	cd web && npm run format

typecheck: ## Strict mypy and tsc
	$(BACKEND) uv run mypy
	cd web && npm run typecheck
