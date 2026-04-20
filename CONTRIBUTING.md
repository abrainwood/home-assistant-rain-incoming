# Contributing to Rain Incoming

Thanks for your interest in contributing! This integration detects approaching rain using RainViewer radar data and provides sensors for Home Assistant automations.

**Quality bar**: resilience, observability, and thorough testing are not optional extras. PRs are expected to maintain the same standard as the existing code.

## AI-assisted development

This project has been developed with substantial AI assistance (Claude Opus and Sonnet) under direct human direction. All architectural decisions, code review, test discipline, and final approvals are made by the human maintainer. Tests, documentation, and production code are reviewed for correctness regardless of how they were written.

Contributions are welcome whether or not you use AI tools. The same DCO sign-off process applies either way - by signing off, you confirm you have the right to submit the contribution, which includes responsibility for any AI-assisted portions.

## Prerequisites

- Python 3.12+ (via [pyenv](https://github.com/pyenv/pyenv) recommended)
- Docker (for E2E tests and local HA dev instance)
- Git

## Getting started

```bash
git clone https://github.com/abrainwood/home-assistant-rain-incoming.git
cd home-assistant-rain-incoming

# Create virtualenv and install dev dependencies
$(pyenv which python) -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Set up git hooks (blocks push if tests fail)
make hooks

# Run tests
make test          # unit + integration (~75s)
make test-e2e      # E2E against Docker HA (~45s)
```

## Data source

This integration uses the [RainViewer API](https://www.rainviewer.com/api.html) - a free public API with no API key required. Their terms specify personal/educational use with attribution. We're tracking formal permission for open-source distribution.

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
- **No test deletion**: a test can only be replaced if the replacement covers the same behaviour at the same layer. Moving coverage to a different layer needs discussion. "This behaviour no longer exists" is a valid reason to delete - with explanation.
- **Self-documenting code**: names should explain intent. Comments only where the logic genuinely needs explanation.
- **Abstract external dependencies**: all data sources go behind the `RadarProvider` interface. Never call an external API directly from core logic.
- **Keep it simple**: don't add features, abstractions, or error handling beyond what the current task requires.
- **Never silently swallow exceptions**: unexpected exceptions and unexpected HTTP status codes must be logged at WARNING or ERROR. `except Exception: pass` or `except Exception: _LOGGER.debug(...)` is never acceptable.
- **Retry with backoff**: use `http_retry.py` for all HTTP calls. Retry on 429/5xx with exponential backoff. Respect `Retry-After` headers on 429 responses. Log at WARNING if all retries fail.
- **Rate limit awareness**: don't fire bursts of concurrent requests. Pace batches, use connection limits.
- **Log enough for users to diagnose**: include HTTP status, URL or service name, and a human-readable message. Don't log raw tracebacks for expected failure modes.
- **Fail visible, not silent**: missing data, empty renders, stale sensors must appear in the HA system log - not hide behind debug-level logging.

### Test quality rules

- **Tests must validate user experience, not just internal logic.** If unit tests pass but a user would see broken behaviour, the tests are lying. Test that data produces correct visible output, not just that a value is non-zero.
- **Test the tests - inverse validation.** For every E2E and golden data positive test, verify there's an assertion that would FAIL if the pipeline were broken. Feed in data that should fail, and confirm the test does fail. A test that passes with both valid and invalid input isn't testing anything.
- **Golden data must exercise the full pipeline.** Real-world captures must run through ALL stages - QC, compositing, rendering - not just the detection algorithm in isolation. A detection test that skips QC confidence application is testing a fiction.
- **No sleeps in tests.** `time.sleep()` causes flakiness. Poll with a timeout instead - or have the system under test signal readiness via logs, events, or state changes.
- **Flaky test = failing test.** Fix it before moving on. Don't skip, quarantine, or rerun to paper over it.

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

## Pull requests

We follow the [Kubernetes PR best practices](https://github.com/kubernetes/community/blob/main/contributors/guide/pull-requests.md#best-practices-for-faster-reviews):

- **Keep PRs small and focused.** One concern per PR - don't bundle bug fixes with refactors with features. Smaller PRs get faster, more thorough reviews.
- **Validate demand first.** For significant new features, open an issue to discuss the approach before investing effort.
- **Separate concerns.** Bug fix, refactor, and feature work belong in distinct PRs.
- **Test thoroughly.** Very few PRs should touch code without touching tests.
- **Document your reasoning.** Non-obvious decisions need comments. Commit bodies should explain *why*, not *what*.

All PRs require a green CI build and at least one approving review before merge.

## Sign your commits (DCO)

This project uses the [Developer Certificate of Origin](https://developercertificate.org/). Every commit must include a `Signed-off-by:` line proving you have the right to contribute the work under this project's licence.

The easiest way to sign off is to use `git commit -s` (or `--signoff`):

```bash
git commit -s -m "Add support for Lake Margaret radar coverage"
```

For an existing commit:

```bash
git commit -s --amend --no-edit
```

For multiple unsigned commits in a branch:

```bash
git rebase -i --signoff main
```

The DCO GitHub Actions check will block merges with unsigned commits. By signing off, you affirm:

1. The contribution is your own work, OR
2. The contribution is based on prior work covered by an appropriate open source licence and you have the right to submit it under that licence, OR
3. The contribution was provided to you by someone who certified one of the above and you are not modifying it.

The full DCO text is at [developercertificate.org](https://developercertificate.org/). It's a short, plain-English statement.

## Commit messages

- Start with what changed, not what you did ("Add cell tracking" not "I added cell tracking")
- Keep the first line under 72 characters
- Use the body for why, not what (the diff shows what)
- Reference the GitHub issue if related (e.g. "Fix arrival time for overhead rain (#42)")

## Licensing intent

This project is released under the MIT Licence (see [LICENSE](LICENSE)). The maintainer reserves the right to dual-licence or relicence the project under other terms (including commercial licences) in the future.

By contributing under the DCO, you acknowledge that your contributions may be included in such relicensed versions of the project. Forks and non-commercial redistribution under the existing MIT terms are explicitly welcome regardless of any future relicensing of this repository - the MIT version of any commit you've seen will always remain available under MIT.

If you'd prefer your contribution NOT be part of any future relicensed version, please open a discussion before submitting the PR.

## Translations

We welcome translations for the config flow and sensor UI. Currently only English is provided.

### How to add a translation

1. Copy `custom_components/rain_incoming/translations/en.json` to a new file named with the [BCP 47 language tag](https://www.iana.org/assignments/language-subtag-registry/language-subtag-registry), e.g. `de.json` for German, `fr.json` for French, `ja.json` for Japanese
2. Translate the values (not the keys) in the new file
3. Do NOT modify `strings.json` - it's the development source and must stay in sync with `en.json`
4. Submit a PR with just the new translation file

### What to translate

The file contains:
- **Config flow**: setup wizard title, description, field labels, error messages, abort messages
- **Options flow**: settings page title and field labels
- **Selector labels**: map style names

### Guidelines

- Keep translations concise - HA UI space is limited
- Preserve `{placeholder}` variables exactly as they appear
- Error messages should be actionable ("must be between 20 and 60" not just "invalid")
- Don't translate technical terms that are well-known in the target language (e.g. "RainViewer", "CARTO", "ESRI")

### Testing your translation

Set your HA instance language to the target language in **Settings > System > General > Language**, then navigate to the Rain Incoming integration setup flow and options flow to verify all strings render correctly.

## References

- [RainViewer API documentation](https://www.rainviewer.com/api.html)
- [RainViewer colour schemes](https://www.rainviewer.com/api/color-schemes.html) - we use scheme 2 (Universal Blue), the only available scheme
