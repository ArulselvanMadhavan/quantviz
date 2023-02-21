open Torch

val amax_mse
  :  Tensor.t
  -> x_max:float
  -> num_mantissa_bits:int
  -> float array * float array
