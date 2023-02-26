open Torch
open Base

let find_max t amax_type =
  let open Tensor in
  match amax_type with
  | Amax_type.Tensor -> maximum t
  | Amax_type.Channel dim ->
    let xs = split t ~split_size:1 ~dim |> List.map ~f:maximum in
    let xs = List.map xs ~f:(Tensor.unsqueeze ~dim:0) in
    stack xs ~dim
;;

let quantize device amax_type names_and_tensors ~bits =
  let module T = Tensor in
  let two = T.of_int0 ~device 2 in
  let n_minus_1 = T.of_int0 ~device Int.(bits - 1) in
  let f (_ttype, t) =
    let t = T.to_ t ~device in
    let t = T.abs_ t in
    let amax_t = find_max t amax_type in
    let max_bound = T.pow two ~exponent:n_minus_1 in
    let scales = T.(amax_t / max_bound) in
    let intq = T.(round (t / scales)) in
    let xq = T.(intq * scales) in
    let meandims = List.init (List.length (T.shape t)) ~f:Fn.id in
    Mse.calc_sqnr t xq meandims
  in
  let mses = List.map names_and_tensors ~f in
  List.iter mses ~f:T.print;
  ()
;;
