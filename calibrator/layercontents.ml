open Torch
open Base

exception UnknownLayerInfo of string

type t =
  { inputs : (string, Tensor.t) Hashtbl.t
  ; layer_variables : (string, Tensor.t) Hashtbl.t
  ; outputs : (string, Tensor.t) Hashtbl.t
  }
[@@deriving make]

let default () =
  let make_ht () = Hashtbl.create (module String) in
  make_t ~inputs:(make_ht ()) ~layer_variables:(make_ht ()) ~outputs:(make_ht ()) ()
;;

let update info_type contents t =
  let t = Option.value t ~default:(default ()) in
  let add_contents ht =
    List.iter contents ~f:(fun (name, tensor) ->
      Hashtbl.update ht name ~f:(fun _ -> tensor));
    ht
  in
  match info_type with
  | "inputs" -> { t with inputs = add_contents t.inputs }
  | "outputs" -> { t with outputs = add_contents t.outputs }
  | "layer_variables" -> { t with layer_variables = add_contents t.layer_variables }
  | _ -> raise (UnknownLayerInfo info_type)
;;
