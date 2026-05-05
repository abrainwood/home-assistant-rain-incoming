# 1. Record architecture decisions

Date: 2026-05-05

## Status

Accepted

## Context

We need a lightweight, durable way to record the architectural decisions made on this project — the kind of decisions that are hard to reverse and that future contributors (and AI agents) need to understand without re-deriving the reasoning from code.

## Decision

We will record architectural decisions as ADRs (Architecture Decision Records) using the format described by Michael Nygard in [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

ADRs live in `docs/adr/` as numbered markdown files (`NNNN-title-in-kebab-case.md`). Each ADR has: Status, Context, Decision, Consequences.

The `improve-codebase-architecture`, `diagnose`, and `tdd` skills read this directory before proposing changes in the affected area, and surface ADR conflicts explicitly rather than silently overriding them.

## Consequences

- Future architectural changes write a new ADR rather than amending the code-only.
- Decisions are searchable in plain text; no external tool dependency.
- A reader can reconstruct the design rationale by reading `docs/adr/` in order.
