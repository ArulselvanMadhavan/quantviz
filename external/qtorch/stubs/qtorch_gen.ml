let () =
  let fmt file = Format.formatter_of_out_channel (open_out file) in
  let fmt_c = fmt "qtorch_stubs.c" in
  Format.fprintf fmt_c "#include \"qtorch_api.h\"@.";
  Cstubs.write_c fmt_c ~prefix:"caml_" (module Qtorch_bindings.C);
  let fmt_ml = fmt "qtorch_generated.ml" in
  Cstubs.write_ml fmt_ml ~prefix:"caml_" (module Qtorch_bindings.C);
  flush_all ()
