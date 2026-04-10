# Rain Incoming - Developer Guide

HA custom integration that detects approaching rain using RainViewer radar data and exposes binary sensors and arrival-time sensors for Home Assistant automations.

## Quick start

```bash
git clone https://github.com/abrainwood/home-assistant-rain-incoming.git
cd incoming_rain
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run unit + integration
python -m pytest tests/unit tests/integration

# Run E2E (needs Docker)
python -m pytest tests/e2e

# Start a dev HA instance with real RainViewer
make dev   # then open http://localhost:8123 (login: dev / devdevdev)
```

## Architecture

**Data source**: RainViewer API - free, no API key required, public radar tiles.

**Detection pipeline**: cell tracking, not pixel-level thresholding. We track storm cells across frames, estimate velocity, and project where they'll be.

**QC pipeline**: multi-stage quality control before a cell is trusted. Stages: texture analysis, temporal consistency, clutter map comparison, speed sanity, motion consistency. Each stage contributes a confidence score.

**Rendering**: binary threshold mask applied to the final composite - not a raw radar image.

### Key files

```
custom_components/rain_incoming/
  coordinator.py          # DataUpdateCoordinator, fetch scheduling, backoff
  image.py                # Camera entity, image rendering
  http_retry.py           # Retry-with-backoff wrapper for all HTTP calls
  providers/
    base.py               # Abstract RadarProvider / RadarFrame interfaces
    rainviewer.py         # RainViewer API implementation
  radar/
    detector.py           # Full detection pipeline orchestration
    composite.py          # Frame compositing
    filters.py            # Threshold, spatial, temporal filters
    motion.py             # Cell centroid tracking, velocity estimation
    geo.py                # Coordinate math
    qc/                   # Quality control subpackage
      scoring.py          # Aggregate confidence score
      texture.py          # Texture-based clutter detection
      temporal.py         # Temporal consistency checks
      clutter.py          # Clutter map generation
      clutter_map.py      # Persistent clutter map management
      speed_sanity.py     # Physically implausible speed rejection
      motion_consistency.py  # Motion vector consistency checks
      types.py            # QC data types

tests/
  unit/                   # Fast, isolated (~seconds)
  integration/            # In-process HA via pytest-homeassistant-custom-component
  contract/               # Live API contract checks (CI schedule only)
  e2e/                    # Full stack against Docker HA
  fixtures/
    golden_data/          # Real-world radar captures (v1)
    golden_v2/            # Real-world radar captures (v2, multi-location)
  radar_scenarios.py      # Synthetic golden scenarios shared across layers
```

## TDD cycle

Write a failing test, write minimum code to pass, refactor. That's it. If you're fixing a bug, write the failing test first.

**Test layers:**
- **Unit**: happy, sad, and bad paths. Run in seconds. Most tests live here.
- **Integration**: HA-specific wiring - config flow, coordinator lifecycle, sensor entities. Uses `pytest-homeassistant-custom-component`, no Docker.
- **Contract**: hit the real RainViewer API to verify the upstream contract. Run on a CI schedule, not every push.
- **E2E**: Docker HA + mock RainViewer server. Full pipeline from HTTP fetch to sensor state.
- **Golden data**: real-world radar captures run through the complete pipeline. Must exercise ALL stages - QC, compositing, rendering - not just the core algorithm.

## Test rules

- **Fix flaky tests immediately.** A flaky test is a failing test. Don't skip, quarantine, or rerun to paper over it.
- **No sleeps in tests.** `time.sleep()` causes flakiness. Poll with a timeout instead.
- **Never delete a test because it's hard to pass.** A test can only be replaced if the replacement covers the same behaviour at the same layer. "This behaviour no longer exists" is a valid reason to delete - with explanation.
- **Tests must validate user experience, not just internal logic.** If your unit tests pass but a user would see broken behaviour, the tests are lying.
- **Test the tests - inverse validation.** For every E2E and golden data positive test ("rain scenario shows rain"), verify there's a corresponding assertion that would FAIL if the pipeline were broken. Feed in bad data and confirm the test fails. A test that passes with both valid and invalid input isn't testing anything.
- **Golden data must exercise the full pipeline.** Run real captures through ALL transforms, filters, QC stages, and rendering steps - not just the detection algorithm in isolation.
- **Don't change tests while refactoring.** If a test needs to change, stop and explain why before proceeding.

## Code quality

- Names should make code self-documenting. Comments only where the logic genuinely needs explanation.
- Keep it simple - from architecture down to individual functions.
- Abstract all third-party dependencies. External APIs go behind the `RadarProvider` interface. Never call RainViewer (or anything else) directly from core logic.
- Boy scout rule: leave code cleaner than you found it - but only when you have enough understanding to be confident you're not breaking anything.
- Restrict scope to the current task. Don't explore or improve unrelated code.
- Fast dev loop. Any green build should be releasable.

## Resilience requirements

Assume external dependencies will break. Code must be resilient and observable.

- **Never silently swallow exceptions.** Unexpected exceptions and unexpected HTTP status codes must be logged at WARNING or ERROR. `except Exception: pass` or `except Exception: _LOGGER.debug(...)` is never acceptable.
- **Retry with backoff for transient failures.** Use `http_retry.py` for all HTTP calls. Retry on 429/5xx with exponential backoff. Respect `Retry-After` headers on 429 responses. Log at WARNING if all retries fail.
- **Rate limit awareness.** Don't fire bursts of concurrent requests. Pace batches, use connection limits.
- **Log enough for users to diagnose.** Include HTTP status, URL or service name, and a human-readable message. Don't log raw tracebacks for expected failure modes.
- **Fail visible, not silent.** Missing data, empty renders, stale sensors - these must appear in the HA system log, not hide behind debug-level logging.

## Handling unknowns

LLMs are naturally optimistic - bias toward pessimism. If you're unsure about something, say so. If you can verify quickly, do it. If you can't, stop and ask rather than silently assume.
