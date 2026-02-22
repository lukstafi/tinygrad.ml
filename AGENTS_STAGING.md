# Agent Learnings (Staging)

This file collects agent-discovered learnings for later curation into CLAUDE.md / AGENTS.md.

<!-- Entry: task_tinygrad_port-codex | 2026-02-22 -->
### Run Dune Commands Sequentially

Avoid running multiple `dune` commands in parallel in this repo. Parallel invocations can interfere with dune RPC state and fail with `Error: RPC server not running.` Run `dune exec ...` and `dune test` sequentially for reliable results.

<!-- End entry -->
