run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

seed-orders:
	python -m app.seed_orders

seed-analyze:
	python -m app.seed_analyze

compose-dev:
	docker compose -f compose.dev.yaml up -d

compose-dev-down:
	docker compose -f compose.dev.yaml down