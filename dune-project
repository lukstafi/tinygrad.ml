(lang dune 3.0)
(name tinygrad_ml)
(generate_opam_files true)

(package
 (name tinygrad_ml)
 (allow_empty)
 (synopsis "A minimal, educational deep learning framework in OCaml")
 (depends
  (ocaml (>= 4.14))
  dune
  ctypes
  ctypes-foreign)
 (depopts metal cudajit))
