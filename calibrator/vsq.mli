open Torch

val quantize
  :  ?channel_dim:int
  -> Device.t
  -> (string * Tensor.t) list
  -> vsize:int
  -> bits:int
  -> (string * string * float) list

val build_rows : string -> (string * string * float) list -> string list list
val dump_rows : Caml.out_channel -> string list list -> unit
