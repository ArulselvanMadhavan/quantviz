open Torch
open Base

exception UnknownLayerInfo of string

module LayerContents = struct
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

  let lengths t =
    List.length t.inputs, List.length t.layer_variables, List.length t.outputs
  ;;
end

let load_tensors ht filename =
  let contents = Serialize.load_all ~filename in
  let layer_info = Quantviz.Utils.layer_name_and_mem filename in
  let layer_name = List.hd_exn layer_info in
  let info_type = List.last_exn layer_info in
  Hashtbl.update ht layer_name ~f:LayerContents.(update info_type contents)
;;

let () =
  let dir_name = "/nfs/nomster/data/arul/data/artifacts/opt125m/fp32/layers/" in
  let files = Quantviz.Utils.dir_contents dir_name ~ext:"ot" in
  let ht = Hashtbl.create ~size:(List.length files) (module String) in
  List.iter ~f:(load_tensors ht) files;
  Hashtbl.iteri ht ~f:(fun ~key ~data ->
    let l1, l2, l3 = LayerContents.lengths data in
    Stdio.printf "%s|%d|%d|%d\n" key l1 l2 l3);
  ()
;;
