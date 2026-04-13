# Add package specification with build dependencies to dune-project

## Goal

The `dune-project` file currently has no `(package ...)` stanza, which means dune cannot generate an `.opam` file and `opam pin`/`opam install` workflows fail. Adding the package stanza with correct dependency declarations enables standard opam integration.

## Acceptance Criteria

- `dune-project` contains a `(package ...)` stanza for `tinygrad_ml` with:
  - Required build dependencies: `ctypes`, `ctypes-foreign`
  - Optional dependencies (depopts): `metal`, `cudajit`
  - A synopsis and description
- Running `dune build` still succeeds
- An opam file is generated (or can be generated via `dune build @install`)

## Context

**Current state:** `dune-project` contains only `(lang dune 3.0)` and `(name tinygrad_ml)`.

**Dependency map** (from `lib/dune`):

| Library | Dependencies |
|---------|-------------|
| `tinygrad_ml` (main) | `unix`, `str`, `ctypes`, `ctypes-foreign` + conditional optional backends |
| `tinygrad_ml_metal` (optional) | `ctypes`, `ctypes-foreign`, `metal` |
| `tinygrad_ml_cuda` (optional) | `ctypes`, `ctypes-foreign`, `cudajit.cuda`, `cudajit.nvrtc` |

`unix` and `str` ship with the OCaml compiler and do not need opam dependency entries. The `metal` and `cudajit` packages are optional backends gated by `(optional)` library stanzas and `(select ...)` clauses.

**Key files:**
- `/dune-project` -- target file
- `/lib/dune` -- source of library dependency declarations
- `/test/dune` -- tests depend only on `tinygrad_ml` and stdlib modules

## Approach

*Suggested approach -- agents may deviate if they find a better path.*

Add a `(package ...)` stanza to `dune-project` after the existing `(name ...)` line:

```dune
(package
 (name tinygrad_ml)
 (synopsis "A minimal, educational deep learning framework in OCaml")
 (depends
  (ocaml (>= 4.14))
  dune
  ctypes
  ctypes-foreign)
 (depopts metal cudajit))
```

The `(depopts ...)` field declares optional dependencies that correspond to the `(optional)` library stanzas in `lib/dune`.

## Scope

**In scope:** Adding the `(package ...)` stanza to `dune-project` with correct required and optional dependencies.

**Out of scope:** Creating a standalone `.opam` file (dune generates this from the package stanza), adding CI opam workflows, publishing to the opam repository.
