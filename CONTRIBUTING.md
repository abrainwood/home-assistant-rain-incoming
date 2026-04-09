# Contributing to Incoming Rain

Thanks for your interest in contributing! This integration detects approaching rain using RainViewer radar data and provides sensors for Home Assistant automations.

## Prerequisites

- Python 3.12+ (via [pyenv](https://github.com/pyenv/pyenv) recommended)
- Docker (for E2E tests and local HA dev instance)
- Git

## Getting started

```bash
git clone <repo-url>
cd incoming_rain

# Create virtualenv and install dev dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Set up git hooks (blocks push if tests fail)
git config core.hooksPath .githooks

# Run tests
make test          # unit + integration (~75s)
make test-e2e      # E2E against Docker HA (~45s)
```

## Data source

This integration uses the [RainViewer API](https://www.rainviewer.com/api.html) - a free public API with no API key required. Their terms specify personal/educational use with attribution. We're tracking formal permission for open-source distribution.

## Project structure

```
custom_components/incoming_rain/
  providers/
    base.py          # Abstract interfaces (RadarProvider, RadarFrame)
    rainviewer.py     # RainViewer API implementation
  radar/
    filters.py        # Threshold, spatial, temporal filters
    motion.py         # Cell centroid tracking, velocity estimation
    detector.py       # Full detection pipeline
  binary_sensor.py    # Rain Incoming sensor (on/off)
  sensor.py           # Rain Arrival Time sensor (timestamp)
  coordinator.py      # DataUpdateCoordinator with backoff
  config_flow.py      # UI configuration flow
  const.py            # All constants and thresholds

tests/
  unit/               # Fast, no external deps (~1s)
  integration/        # In-process HA via pytest-homeassistant-custom-component (~75s)
  contract/           # Live API contract checks (scheduled in CI)
  e2e/                # Full stack against Docker HA (~45s)
  radar_scenarios.py  # Golden test data shared across test layers
```

## Test layers

We maintain four test layers. Each has a different purpose - don't skip layers.

### Unit tests (`make test-unit`)

Fast, isolated, no network. The vast majority of tests live here. Cover happy, sad, and bad paths. These should run in under 2 seconds.

### Integration tests (`make test-integration`)

Test HA-specific behaviour: config flow, sensor entity creation, coordinator lifecycle. Use `pytest-homeassistant-custom-component` to spin up an in-process HA instance. No Docker needed.

### Contract tests (`make test-contract`)

Hit the real RainViewer API to verify the upstream contract hasn't changed. These tests don't test our logic - they verify that the API still returns JSON manifests with the expected structure, that tiles are valid PNGs at zoom 7, etc. Run on a schedule in CI (not on every push) so builds don't flake on transient API issues.

### E2E tests (`make test-e2e`)

Full stack: Docker HA container + mock RainViewer server. Tests the entire pipeline from HTTP fetch through to sensor state updates. The mock server supports scenario switching (no rain / approaching / overhead) so we can verify detection results without waiting for real weather.

## Code expectations

- **TDD**: write the test first, then the code to pass it. If you're fixing a bug, write the failing test that reproduces it before fixing.
- **No test deletion**: a test can only be replaced if the replacement covers the same behaviour. Moving coverage to a different layer needs discussion.
- **Self-documenting code**: names should explain intent. Comments only where the logic genuinely needs explanation.
- **Abstract external dependencies**: all data sources go behind the `RadarProvider` interface. Never call an external API directly from core logic.
- **Keep it simple**: don't add features, abstractions, or error handling beyond what the current task requires.

## Running the dev HA instance

For manual testing with a real HA UI:

```bash
make dev               # start HA with real RainViewer
# open http://localhost:8123
# login: dev / devdevdev

make dev-restart       # recreate container after code changes
make dev-logs          # tail HA logs
make dev-stop          # shut down
```

The dev instance and E2E tests use separate Docker containers, volumes, and ports so they don't interfere with each other:

| | Dev | E2E |
|---|---|---|
| Container | `ha-dev` | `ha-e2e` |
| Volume | `ha-dev-config` | `ha-e2e-config` |
| Port | 8123 | 18123 |

`make dev-restart` recreates the container (not just stop/start), so any stale environment from a previous mock session is cleared.

## Useful make targets

| Target | What it does |
|--------|-------------|
| `make test` | Unit + integration tests |
| `make test-unit` | Unit tests only |
| `make test-integration` | Integration tests only |
| `make test-e2e` | E2E tests (starts Docker) |
| `make dev` | Start HA dev instance |
| `make dev-restart` | Restart after code changes |
| `make dev-stop` | Stop dev instance |
| `make dev-logs` | Tail HA logs |

## Commit messages

- Start with what changed, not what you did ("Add cell tracking" not "I added cell tracking")
- Keep the first line under 72 characters
- Use the body for why, not what (the diff shows what)

## References

- [RainViewer API documentation](https://www.rainviewer.com/api.html)
- [RainViewer colour schemes](https://www.rainviewer.com/api/color-schemes.html) - we use scheme 2 (Universal Blue), the only available scheme
