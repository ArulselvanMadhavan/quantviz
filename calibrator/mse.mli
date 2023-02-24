open Torch

val amax_mse : ?channel_dim:int -> Tensor.t -> num_mantissa_bits:int -> Tensor.t

val quantize_to_fp8
  :  ?channel_dim:int
  -> Tensor.t
  -> Tensor.t
  -> num_mantissa_bits:int
  -> Tensor.t

val calc_mse : Tensor.t -> Tensor.t -> int list -> Tensor.t
val calc_sqnr : Tensor.t -> Tensor.t -> int list -> Tensor.t
