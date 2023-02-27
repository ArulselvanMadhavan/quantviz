open Torch

val quantize
  :  ?channel_dim:int
  -> Device.t
  -> (string * Tensor.t) list
  -> vsizes:int list
  -> tensor_bits:int list
  -> scale_bits:int list
  -> (string * string * float * int) list

val build_rows : string -> (string * string * float * int) list -> string list list
val dump_rows : Caml.out_channel -> string list list -> unit
