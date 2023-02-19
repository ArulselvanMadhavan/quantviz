open Torch
open Base
open Calibrator

let load_tensors ht filename =
  let contents = Serialize.load_all ~filename in
  let layer_info = Quantviz.Utils.layer_name_and_mem filename in
  let layer_name = List.hd_exn layer_info in
  let info_type = List.last_exn layer_info in
  Hashtbl.update ht layer_name ~f:Layercontents.(update info_type contents)
;;

let () =
  let dir_name = "/nfs/nomster/data/arul/data/artifacts/opt125m/fp32/layers/" in
  let files = Quantviz.Utils.dir_contents dir_name ~ext:"ot" in
  let ht = Hashtbl.create ~size:(List.length files) (module String) in
  List.iter ~f:(load_tensors ht) files;
  Hashtbl.iteri ht ~f:(fun ~key ~data ->
    let l1, l2, l3 = Layercontents.lengths data in
    Stdio.printf "%s|%d|%d|%d\n" key l1 l2 l3);
  ()
;;
