open Base
open Torch
(* Assumes only positive values *)

let maxval_span_length = 1    (* calibration lookup points *)

let quantize_to_fp8 ?channel_dim t maxval ~num_mantissa_bits =
  let open Tensor in
  let device = device t in
  let one = of_float0 ~device 1. in
  let two = of_float0 ~device 2. in
  let seven = of_int0 ~device 7 in
  let num_mantissa_bits = of_int0 ~device num_mantissa_bits in
  let exp_bits = seven - num_mantissa_bits in
  (* maxval.shape == t.shape if channel_dim is not None *)
  let f _ dim =
    let shape =
      List.init (List.length (shape t)) ~f:(fun i -> if Int.(i = dim) then -1 else 1)
    in
    reshape maxval ~shape
  in
  let maxval = Option.fold channel_dim ~init:maxval ~f in
  (* compute bias *)
  let exp_val = pow two ~exponent:exp_bits - one in
  let log2maxval = log2 maxval in
  let max_mantissa = log2 (two - pow two ~exponent:(neg num_mantissa_bits)) in
  let bias = exp_val + max_mantissa - log2maxval in
  (* let bias = pow two ~exponent:(exp_bits - one) in *)
  let x_clipped = max t (neg maxval) in
  let x_clipped = min x_clipped maxval in
  let log_scales = floor (log2 (abs x_clipped + bias)) in
  let scale_max = maximum log_scales |> to_float0_exn in
  let log_scales = clamp log_scales ~min:(Scalar.f 1.) ~max:(Scalar.f scale_max) in
  let scales = log_scales - num_mantissa_bits - bias in
  let scales = pow two ~exponent:scales in
  round (div x_clipped scales) * scales
;;

let calc_mse ?channel_dim t xfp meandims =
  let open Tensor in
  let meandims =
    Option.fold channel_dim ~init:meandims ~f:(fun acc cdim ->
      List.filter acc ~f:(Int.( <> ) cdim))
  in
  let mse = pow (t - xfp) ~exponent:(of_int0 2 ~device:(device t)) in
  mean_dim mse ~dim:(Some meandims) ~keepdim:false ~dtype:(kind mse)
;;

let calc_sqnr t qt meandims =
  let open Tensor in
  let two = of_int0 ~device:(device t) 2 in
  let noise = pow (qt - t) ~exponent:two in
  let signal = pow t ~exponent:two in
  let noise = mean_dim noise ~dim:(Some meandims) ~keepdim:false ~dtype:(kind noise) in
  let signal = mean_dim signal ~dim:(Some meandims) ~keepdim:false ~dtype:(kind signal) in
  let ratio = div signal noise in
  let ratio = log10 ratio in
  mul_scalar ratio (Scalar.i 10)
;;

let calc_linspaces ?channel_dim t =
  let max_as_float t = Tensor.maximum t |> Tensor.to_float0_exn in
  let mul_factor x_max mul = Scalar.f (mul *. x_max) in
  let linspace (t, x_max) =
    Tensor.linspace
      ~start:(mul_factor x_max 1.0) (*  *)
      ~end_:(mul_factor x_max 1.0)
      ~steps:maxval_span_length
      ~options:Tensor.(kind t, device t)
  in
  let init () = linspace (t, max_as_float t) |> Tensor.unsqueeze ~dim:1 in
  let handle_cdim _ dim () =
    let splits = Tensor.split t ~split_size:1 ~dim in
    let x_maxs = List.map splits ~f:max_as_float in
    let ts = List.zip_exn splits x_maxs |> List.map ~f:linspace in
    Tensor.stack ts ~dim
  in
  Option.fold channel_dim ~init ~f:handle_cdim ()
;;

let amax_mse ?channel_dim t ~num_mantissa_bits =
  let open Tensor in
  let linspaces = calc_linspaces ?channel_dim t in
  let ndims = List.length (shape t) in
  let meandims = List.init ndims ~f:Fn.id in
  let i = ref 0 in
  let mses = Tensor.to_ (Tensor.zeros_like linspaces) ~device:(device t) in
  let mses = ref mses in
  while !i < maxval_span_length do
    let maxval = select linspaces ~dim:0 ~index:!i in
    let xfp = quantize_to_fp8 t maxval ~num_mantissa_bits in
    let mse = calc_mse ?channel_dim t xfp meandims in
    let index = Tensor.of_int0 ~device:(device t) !i in
    mses
      := Tensor.index_put_
           !mses
           ~indices:[ Some index; None ]
           ~values:mse
           ~accumulate:false;
    Caml.Gc.full_major ();
    i := Int.(!i + 1)
  done;
  let mses = !mses in
  let best_mse = Tensor.argmin mses ~dim:(Some 0) ~keepdim:false in
  let num_channels = Tensor.shape linspaces |> List.last_exn in
  let maxval = Array.create ~len:num_channels 0. in
  let maxval =
    Array.mapi maxval ~f:(fun i _ ->
      let best_mse_idx = Tensor.( .%[] ) best_mse i in
      let mse_pos = Tensor.( .%.{} ) linspaces [ best_mse_idx; i ] in
      mse_pos)
  in
  Tensor.of_float1 ~device:(device t) maxval
;;
