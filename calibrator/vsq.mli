open Torch

val quantize : Device.t -> Amax_type.t -> (string * Tensor.t) list -> bits:int -> unit
