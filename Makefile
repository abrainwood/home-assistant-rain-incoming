.PHONY: test test-unit test-integration test-contract test-e2e dev dev-mock dev-stop dev-logs dev-restart

# --- Tests ---

test: test-unit test-integration
	@echo "\nAll tests passed"

test-unit:
	.venv/bin/pytest tests/unit/ -q

test-integration:
	.venv/bin/pytest tests/integration/ -q

test-contract:
	.venv/bin/pytest tests/contract/ -v --override-ini="asyncio_mode=auto" -p no:socket

test-e2e:
	.venv/bin/pytest tests/e2e/ -v --tb=short -x

# --- Dev environment ---

dev:
	docker compose -f docker-compose.dev.yml up -d
	.venv/bin/python scripts/dev-setup.py

dev-mock:
	.venv/bin/python scripts/dev-setup.py --mock

dev-stop:
	docker compose -f docker-compose.dev.yml down

dev-logs:
	docker logs ha-dev -f --tail 50

dev-restart:
	docker compose -f docker-compose.dev.yml stop
	docker compose -f docker-compose.dev.yml start
