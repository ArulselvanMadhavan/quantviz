open Torch
open Base

exception UnknownLayerInfo of string

type t =
  { inputs : (string * Tensor.t) list
  ; layer_variables : (string * Tensor.t) list
  ; outputs : (string * Tensor.t) list
  }
[@@deriving make]

let default () = make_t ~inputs:[] ~layer_variables:[] ~outputs:[] ()

let update info_type contents t =
  let t = Option.value t ~default:(default ()) in
  match info_type with
  | "inputs" -> { t with inputs = contents }
  | "outputs" -> { t with outputs = contents }
  | "layer_variables" -> { t with layer_variables = contents }
  | _ -> raise (UnknownLayerInfo info_type)
;;

let lengths t = List.length t.inputs, List.length t.layer_variables, List.length t.outputs
