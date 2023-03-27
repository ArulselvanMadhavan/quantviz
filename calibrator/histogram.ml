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
        ~end_:(Scalar.f x_max)
        ~steps:(num_bins + 1)
        ~options:(T Torch_core.Kind.f32, Device.Cpu)
  }
;;

let collect t x = Tensor.histc x ~bins:t.num_bins

let amax_percentile t ~hist ~percentile =
  let hist = Tensor.div hist (Tensor.sum hist) in
  (* let hist = Tensor.div_scalar hist (Scalar.i numel) in *)
  let cdf = Tensor.cumsum hist ~dim:0 ~dtype:(Tensor.kind hist) in
  let idx =
    Tensor.searchsorted
      ~sorted_sequence:cdf
      (Tensor.of_float0 (percentile /. 100.) ~device:(Tensor.device hist))
      ~out_int32:false
      ~side:"left"
      ~right:false
      ~sorter:None
  in
  let idx = Tensor.to_int0_exn idx in
  Tensor.( .%.[] ) t.calib_bin_edges idx
;;

(* amax_mse *)
(* 1. find abs max; 
2. start with per tensor quant - max_val - 1 elem
3. create bins around max_val - 0.1 * max_val to 1.2 * max_val
4. max_val - 111 x 1
5. For each max_val,
6.   quantize tensor
7. find mse
6. find_max_val that has the min quantization error
7. return max_val and quantization_error
*)
