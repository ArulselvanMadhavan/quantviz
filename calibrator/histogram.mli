open Torch
open Base

type t =
  { num_bins : int
  ; calib_bin_edges : Tensor.t
  }

val collect : t -> Tensor.t -> Tensor.t
val make_t : num_bins:int -> x_max:float Scalar.t -> t
val amax_percentile : t -> hist:Tensor.t -> numel:int -> percentile:float -> float
