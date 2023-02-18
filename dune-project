(lang dune 3.6)

(name quantviz)

(generate_opam_files true)

(source
 (github ArulselvanMadhavan/ArulselvanMadhavan))

(authors "Arulselvan Madhavan")

(maintainers "arulselvan1234@gmail.com")

(license MIT)

(documentation https://url/to/documentation)

(package
 (name quantviz)
 (synopsis "A short synopsis")
 (description "A longer description")
 (depends
  (ocaml (>= 4.14.1))
  dune
  containers
  ocamlformat
  ocaml-torch
  npy
  fpath
  pyml)
 (tags
  (topics "to describe" your project)))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/dune-files.html#dune-project
