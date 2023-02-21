open Base
open Torch
(* Assumes only positive values *)

let maxval_span_length = 111

let quantize_to_fp8 t maxval ~num_mantissa_bits =
  let open Tensor in
  let device = device t in
  let one = of_float0 ~device 1. in
  let two = of_float0 ~device 2. in
  let seven = of_int0 ~device 7 in
  let num_mantissa_bits = of_int0 ~device num_mantissa_bits in
  let exp_bits = seven - num_mantissa_bits in
  (* compute bias *)
  let exp_val = pow two ~exponent:exp_bits in
  let log2maxval = log2 maxval in
  let temp = log2 (two - pow two ~exponent:(neg num_mantissa_bits)) in
  let bias = exp_val - log2maxval + temp - one in
  let x_clipped = max t (neg maxval) in
  let x_clipped = min x_clipped maxval in
  let log_scales = floor (log2 (abs x_clipped + bias)) in
  let scale_max = maximum log_scales |> to_float0_exn in
  let log_scales = clamp log_scales ~min:(Scalar.f 1.) ~max:(Scalar.f scale_max) in
  let scales = log_scales - num_mantissa_bits - bias in
  let scales = pow two ~exponent:scales in
  round (div x_clipped scales) * scales
;;

let amax_mse t ~x_max ~num_mantissa_bits =
  let open Tensor in
  let mul_factor mul = Scalar.f (mul *. x_max) in
  let lsp =
    linspace
      ~start:(mul_factor 0.1)
      ~end_:(mul_factor 1.2)
      ~steps:maxval_span_length
      ~options:(kind t, device t)
  in
  let linspaces = unsqueeze lsp ~dim:1 in
  let ndims = List.length (shape t) in
  let meandims = List.init ndims ~f:Fn.id in
  let i = ref 0 in
  let mses = Tensor.to_ (Tensor.zeros_like linspaces) ~device:(device t) in
  let mses = ref mses in
  while !i < maxval_span_length do
    Stdio.printf "Mse:%d\n" !i;
    Stdio.Out_channel.flush Stdio.stdout;
    let maxval = select linspaces ~dim:0 ~index:!i in
    let xfp = quantize_to_fp8 t maxval ~num_mantissa_bits in
    let mse = pow (t - xfp) ~exponent:(of_int0 2) in
    let mse = mean_dim mse ~dim:(Some meandims) ~keepdim:false ~dtype:(kind mse) in
    mses := Tensor.put_ !mses ~index:(of_int0 !i) ~source:mse ~accumulate:false;
    Caml.Gc.full_major ();
    i := Int.(!i + 1)
  done;
  let mses = !mses in
  let best_mse = Tensor.argmin mses ~dim:(Some 0) ~keepdim:true in
  let num_channels = Tensor.shape linspaces |> List.last_exn in
  let maxval = Array.create ~len:num_channels 0. in
  let maxval =
    Array.mapi maxval ~f:(fun i _ ->
      let best_mse_idx = Tensor.( .%[] ) best_mse i in
      let mse_pos = Tensor.( .%.{} ) linspaces [ best_mse_idx; i ] in
      let mse_val = Tensor.( .%.{} ) mses [ best_mse_idx; i ] in
      mse_pos, mse_val)
  in
  Array.unzip maxval
;;
