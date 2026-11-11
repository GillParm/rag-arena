# Common tasks. Run `make help` for the list.
SHELL := /bin/bash
BACKEND := cd backend &&
CLI := $(BACKEND) uv run rag-arena

.PHONY: help hooks install db-up db-down test test-db lint fmt typecheck corpus index evalset bench report api web up down screenshots

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
	$(BACKEND) uv run pytest -m "not db and not live"

test-db: db-up ## Run the Postgres integration tests
	$(BACKEND) uv run pytest -m db

corpus: ## Download the pinned FastAPI docs and check the licence
	$(CLI) corpus download

index: db-up ## Chunk, embed and index the corpus with all three chunkers
	$(CLI) db migrate && $(CLI) index build

evalset: ## Write and filter the eval set with Claude (costs about $1.20)
	$(CLI) eval build

bench: ## Run the full benchmark (Gemini free tier + about $5 of Claude judging)
	$(CLI) bench run --reranker-variants

report: ## Rebuild results/runs/main summaries from the raw records
	$(CLI) bench report

api: ## Run the API on port 8300 with reload
	$(CLI) serve --reload

web: ## Run the web app on port 3300
	cd web && npm run dev

up: ## Build and start Postgres, the API and the web app in Docker
	docker compose up -d --build

down: ## Stop the Docker stack
	docker compose down

screenshots: ## Take README screenshots (API and web must be running)
	cd web && npm run screenshots

lint: ## Lint backend and web
	$(BACKEND) uv run ruff check . && uv run ruff format --check .
	cd web && npm run lint && npm run format:check

fmt: ## Format backend and web
	$(BACKEND) uv run ruff format . && uv run ruff check --fix .
	cd web && npm run format

typecheck: ## Strict mypy and tsc
	$(BACKEND) uv run mypy
	cd web && npm run typecheck
