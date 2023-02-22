open Torch

val amax_mse
  :  Tensor.t
  -> x_max:float
  -> num_mantissa_bits:int
  -> (float * (float * float)) array

val quantize_to_fp8 : Tensor.t -> Tensor.t -> num_mantissa_bits:int -> Tensor.t
val calc_mse : Tensor.t -> Tensor.t -> int list -> Tensor.t
val calc_sqnr : Tensor.t -> Tensor.t -> int list -> Tensor.t
