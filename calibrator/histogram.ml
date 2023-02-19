open Base
open Torch

type t =
  { num_bins : int
  ; calib_bin_edges : Tensor.t
  }
[@@deriving make]

let make_t ~num_bins ~x_max =
  { num_bins
  ; calib_bin_edges =
      Tensor.linspace
        ~start:(Scalar.f 0.)
        ~end_:x_max
        ~steps:(num_bins + 1)
        ~options:(T Torch_core.Kind.f32, Device.Cpu)
  }
;;

let collect t x = Tensor.histc x ~bins:t.num_bins
