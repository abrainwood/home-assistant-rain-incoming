# Issue tracker: GitHub

Issues and PRDs for the **rain_incoming Home Assistant integration** live as GitHub issues at `abrainwood/home-assistant-rain-incoming` (public). Use the `gh` CLI for all operations.

## Repo routing

This repo has a **sister repo**: `abrainwood/rain-incoming-backtester` (private). Route issues by subject:

- **Integration concerns** (detection, QC, sensors, HA wiring, rendering, providers, config flow): file in `abrainwood/home-assistant-rain-incoming` (this repo).
- **Backtester concerns** (replay engine, verifier, capture collector, sweep scripts, manifest tooling, observation correlation): file in `abrainwood/rain-incoming-backtester`.
- **Issues that touch both** (e.g. an integration change driven by a backtest result): file in the integration repo and link the backtester counterpart with `abrainwood/rain-incoming-backtester#nnn`.

Always confirm `git remote -v` before creating an issue — `gh` infers the repo from the cwd, so being in the wrong clone silently posts to the wrong tracker.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

## When a skill says "publish to the issue tracker"

Create a GitHub issue in the appropriate repo (see Repo routing above).

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.
