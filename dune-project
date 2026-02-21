(lang dune 3.10)

(name tinygrad_ml)
(generate_opam_files true)

(source
 (github lukstafi/tinygrad.ml))

(authors "tinygrad.ml contributors")
(maintainers "tinygrad.ml contributors")
(license MIT)

(package
 (name tinygrad_ml)
 (synopsis "Educational OCaml port of tinygrad middle layer")
 (description "Minimal lazy tensor execution stack with CPU C, CUDA, and Metal backends."))
