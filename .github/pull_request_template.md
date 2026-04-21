<!--
SPDX-FileCopyrightText: 2026 bartzbeielstein
SPDX-License-Identifier: AGPL-3.0-or-later
-->

## Summary

<!-- What does this change do, and why? One or two sentences. -->

## Type of change

<!-- Match the Conventional-Commits type of the commit that merges this PR. -->

- [ ] `feat` — new feature
- [ ] `fix` — bug fix
- [ ] `perf` — performance
- [ ] `refactor` — structural change, no behaviour change
- [ ] `docs` — documentation only
- [ ] `test` — tests only
- [ ] `chore` — build, deps, CI
- [ ] `feat!` / `fix!` — breaking change (MAJOR bump)

## Safety-critical checklist

All PRs that change code under `src/spotforecast2_safe/` must pass the four checks below.
Tick every box or explain in the PR body why the check does not apply.

- [ ] **Determinism.** Same input produces the same bit-level output; no new unseeded RNG, hash-order dependency, or unordered parallelism.
- [ ] **Fail-safe semantics.** Invalid input (NaN, wrong dtype, missing index, network failure) raises an explicit `ValueError` / `TypeError` / subclass of `spotforecast2_safe.exceptions` rather than being silently repaired.
- [ ] **Prohibited-dependency guard.** `uv.lock` still resolves cleanly and contains none of `plotly`, `matplotlib`, `spotoptim`, `optuna`, `torch`, `tensorflow`. Verified by `uv run pytest tests/test_prohibited_dependencies.py` and the `prohibited-deps` CI job.
- [ ] **REUSE / SPDX.** Every new source file carries the SPDX header; `uv run reuse lint` passes.

## Threat-model update

This PR changes the *network-facing attack surface* (requests sent, responses parsed, credentials handled, on-disk cache format, authentication boundaries, or downstream consumers of externally-fetched data) if it touches any of the following:

- `src/spotforecast2_safe/downloader/` (ENTSO-E client)
- `src/spotforecast2_safe/weather/` (Open-Meteo client)
- `src/spotforecast2_safe/data/fetch_data.py`
- any new module that opens a socket, issues an HTTP request, or reads a credential from the environment

- [ ] This PR does **not** change the network-facing attack surface. *(Skip the rest of this section.)*
- [ ] This PR **does** change the network-facing attack surface. I have updated the STRIDE table in the module docstring of every affected network-facing file in the same commit, covering all six categories (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege). New data flows added to the table name the countermeasure and the source file that implements it.

The rule is anchored in `CONTRIBUTING.md` §"Threat-model update rule" and cited in the compliance mapping against IEC 62443-4-1 SR-1 / SR-2 and EU AI Act Article 9.

## Tests

- [ ] `uv run pytest tests/ -v` passes locally.
- [ ] Coverage for changed code is ≥ 80 % (gate in CI).
- [ ] New public symbols have an executable docstring example covered by `tests/test_docstring_examples_*.py`.

## Breaking changes

<!-- If this PR bumps the major version (feat! / fix!), describe the migration path. Otherwise write "None". -->

## Linked issues

<!-- Closes #123, Refs #456, or "None". -->
