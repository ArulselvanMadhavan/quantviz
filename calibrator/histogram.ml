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

let amax_percentile t ~hist ~numel ~percentile =
  let hist = Tensor.div_scalar hist (Scalar.i numel) in
  let cdf = Tensor.cumsum hist ~dim:0 ~dtype:(Tensor.kind hist) in
  let idx =
    Tensor.searchsorted
      ~sorted_sequence:cdf
      (Tensor.of_float0 (percentile /. 100.))
      ~out_int32:false
      ~side:"left"
      ~right:false
      ~sorter:None
  in
  let idx = Tensor.to_int0_exn idx in
  Tensor.( .%.[] ) t.calib_bin_edges idx
;;
