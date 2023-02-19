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

let numel t = List.fold (Tensor.size t) ~init:1 ~f:Int.( * )

let () =
  let dir_name = "/nfs/nomster/data/arul/data/artifacts/opt125m/fp32/layers/" in
  let files = Quantviz.Utils.dir_contents dir_name ~ext:"ot" in
  let ht = Hashtbl.create ~size:(List.length files) (module String) in
  List.iter ~f:(load_tensors ht) files;
  Hashtbl.iteri ht ~f:(fun ~key ~data ->
    let l1, l2, l3 = Layercontents.lengths data in
    List.iter data.outputs ~f:(fun (_, t) ->
      let t = Tensor.abs_ t in
      let ndims = List.length (Tensor.shape t) in
      let x_max = Tensor.max_values t ~dim:(List.init ndims ~f:Fn.id) ~keepdim:false in
      let x_max = Tensor.to_bigarray x_max ~kind:Bigarray.Float32 in
      let x_max = Bigarray.array0_of_genarray x_max in
      let x_max = Bigarray.Array0.get x_max in
      let x_max = Scalar.f x_max in
      let h_calib = Calibrator.Histogram.make_t ~num_bins:2048 ~x_max in
      let hist = Calibrator.Histogram.collect h_calib t in
      Stdio.printf "Tensor:%d|%d\n" (numel hist) (numel h_calib.calib_bin_edges));
    Stdio.printf "%s|%d|%d|%d\n" key l1 l2 l3);
  ()
;;
