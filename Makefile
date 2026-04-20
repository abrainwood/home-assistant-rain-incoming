.PHONY: test test-unit test-integration test-contract test-e2e hooks dev dev-mock dev-stop dev-logs dev-restart

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

# --- Git hooks ---

hooks:
	git config core.hooksPath .githooks
	@echo "Git hooks installed (pre-push runs make test)"

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
	docker compose -f docker-compose.dev.yml down
	docker compose -f docker-compose.dev.yml up -d

# --- Volume safety ---
# Prevent accidental deletion of dev HA volumes. These contain config,
# auth tokens, and integration state that take time to recreate.

dev-nuke:
	@echo "WARNING: This will DELETE all HA dev data (config, auth, integrations)."
	@echo "Volume: ha-dev-config"
	@read -p "Type 'yes' to confirm: " confirm && [ "$$confirm" = "yes" ] || (echo "Aborted."; exit 1)
	docker compose -f docker-compose.dev.yml down -v
	@echo "Dev volumes removed. Run 'make dev' to start fresh."
