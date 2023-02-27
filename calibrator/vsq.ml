open Torch
open Base

let numel = List.fold ~init:1 ~f:( * )

let find_max t amax_type =
  let open Tensor in
  match amax_type with
  | Amax_type.Tensor -> maximum t
  | Amax_type.Channel dim ->
    let xs = split t ~split_size:1 ~dim |> List.map ~f:maximum in
    let xs = List.map xs ~f:(Tensor.unsqueeze ~dim:0) in
    stack xs ~dim
  | Amax_type.Vector size ->
    let s = shape t in
    let c = List.last_exn s in
    let j = Int.(c / size) in
    let s = List.drop_last_exn s in
    let s = List.append s [ j; -1 ] in
    let t = reshape t ~shape:s in
    let values, _ = max_dim t ~dim:(-2) ~keepdim:false in
    values
;;

let build_format ~prefix ~vsize ~n ~m =
  let tensor_format = prefix ^ Int.to_string n in
  let vector_format = "V" ^ Int.to_string vsize in
  let s1_format = "S" ^ Int.to_string m in
  let g_format = "G_fp32" in
  String.concat ~sep:"|" [ vector_format; tensor_format; s1_format; g_format ]
;;

let coarse_quant ?channel_dim device names_and_tensors ~bits =
  let module T = Tensor in
  let two = T.of_int0 ~device 2 in
  let n_minus_1 = T.of_int0 ~device Int.(bits - 1) in
  let amax_type = Amax_type.from_channel_dim channel_dim in
  let f (ttype, t) =
    let t = T.to_ t ~device in
    let t = T.abs_ t in
    let amax_t = find_max t amax_type in
    let max_bound = T.pow two ~exponent:n_minus_1 in
    let scales = T.(amax_t / max_bound) in
    let intq = T.(round (t / scales)) in
    let xq = T.(intq * scales) in
    let meandims = List.init (List.length (T.shape t)) ~f:Fn.id in
    let prefix =
      match ttype with
      | "weight" -> "W"
      | _ -> "A"
    in
    ( ttype
    , prefix ^ Int.to_string bits
    , Mse.calc_sqnr t xq meandims |> Tensor.to_float0_exn
    , numel (T.shape scales) * 4 )
  in
  List.map names_and_tensors ~f
;;

let fine_quant device names_and_tensors ~vsize ~n ~m =
  let m = m in
  let module T = Tensor in
  let two = T.of_int0 ~device 2 in
  let n_minus_1 = T.of_int0 ~device Int.(n - 1) in
  let m_minus_1 = T.of_int0 ~device Int.(m - 1) in
  let f (ttype, t) =
    let t = T.to_ ~device t in
    let t = T.abs_ t in
    let ndim = List.length (T.shape t) - 1 in
    let format, vdim, cut_dim =
      match ttype with
      | "weight" -> build_format ~prefix:"W" ~vsize ~n ~m, ndim + 2, ndim + 1
      | _ -> build_format ~prefix:"A" ~vsize ~n ~m, ndim + 1, ndim + 2
    in
    let vdim = vdim - 1 in
    let cut_dim = cut_dim - 1 in
    (* Replace last dim with 2 dims. So, Subtract 1 dim *)
    let tshape = T.shape t in
    let dims = [| -1; -1 |] in
    dims.(vdim - ndim) <- vsize;
    let tshape = List.drop_last_exn tshape in
    let tshape = List.append tshape (Array.to_list dims) in
    let t = T.view t ~size:tshape in
    let xmax_v, _ = T.max_dim t ~dim:vdim ~keepdim:true in
    let max_bound = T.pow two ~exponent:n_minus_1 in
    let s_v = T.(xmax_v / max_bound) in
    let xq = T.(round (t / s_v)) in
    let s_k, _ = T.max_dim s_v ~dim:cut_dim ~keepdim:true in
    let max_bound = T.pow two ~exponent:m_minus_1 in
    let g_k = T.(s_k / max_bound) in
    let s_q = T.(round (s_v / g_k)) in
    let xq2 = T.(xq * s_q * g_k) in
    let scale_size = numel (T.shape s_q) + (numel (T.shape g_k) * 4) in
    let meandims = List.init (List.length (T.shape t)) ~f:Fn.id in
    ttype, format, Mse.calc_sqnr t xq2 meandims |> Tensor.to_float0_exn, scale_size
  in
  List.map names_and_tensors ~f
;;

let quantize ?channel_dim device names_and_tensors ~vsizes ~tensor_bits ~scale_bits =
  let open List.Let_syntax in
  let f_sqnrs =
    let%bind vsize = vsizes in
    let%bind n = tensor_bits in
        let%bind m = scale_bits in
    fine_quant device names_and_tensors ~vsize ~n ~m
  in
  let c_sqnrs =
    let%bind bits = tensor_bits in
    coarse_quant ?channel_dim device names_and_tensors ~bits
  in
  c_sqnrs @ f_sqnrs
;;

let build_rows lname results =
  let build_row (ttype, format, sqnr, scale_size) =
    [ lname
    ; ttype
    ; format
    ; "-1"
    ; "-1"
    ; "-1"
    ; Float.to_string sqnr
    ; format
    ; Int.to_string scale_size
    ]
  in
  List.map results ~f:build_row
;;

let dump_rows oc rows =
  Csv.write_header oc Csv.calib_columns;
  Csv.write_calib oc (-1, rows)
;;
