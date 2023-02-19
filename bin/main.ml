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

let _numel t = List.fold (Tensor.size t) ~init:1 ~f:Int.( * )

let find_max t =
  let ndims = List.length (Tensor.shape t) in
  let x_max = Tensor.max_values t ~dim:(List.init ndims ~f:Fn.id) ~keepdim:false in
  Tensor.to_float0_exn x_max
;;

let columns = [ "layer"; "bin_start"; "bin_end"; "count"; "type_" ]

let write_row oc row =
  Stdio.Out_channel.output_string oc row;
  Stdio.Out_channel.output_char oc '\n'
;;

let write_header oc =
  let header = String.concat ~sep:"," columns in
  write_row oc header
;;

let write_csv layer_name (data : Layercontents.t) =
  let data_dir = "data" in
  let fname = Option.value_exn (Result.ok (Fpath.of_string data_dir)) in
  let _ = Bos.OS.Dir.create fname in
  let fname = Fpath.add_seg fname layer_name in
  let fname = Fpath.add_ext "csv" fname in
  let f oc _ =
    write_header oc;
    let names_and_tensors =
      [ "outputs", data.outputs
      ; "inputs", data.inputs
      ; "layer_variables", data.layer_variables
      ]
    in
    List.iter names_and_tensors ~f:(fun (lname, nt) ->
      List.iter nt ~f:(fun (tname, t) ->
        let t = Tensor.abs_ t in
        let x_max = Scalar.f (find_max t) in
        let h_calib = Calibrator.Histogram.make_t ~num_bins:2048 ~x_max in
        let counts = Calibrator.Histogram.collect h_calib t in
        let ttype = if String.(lname = "layer_variables") then tname else lname in
        let bins = Tensor.to_float1_exn h_calib.calib_bin_edges in
        let counts = Tensor.to_float1_exn counts in
        Array.iteri counts ~f:(fun idx count ->
          let bin_start = Float.to_string bins.(idx) in
          let bin_end = Float.to_string bins.(idx + 1) in
          let count = Float.to_string count in
          let row = [ layer_name; bin_start; bin_end; count; ttype ] in
          let row = String.concat ~sep:"," row in
          write_row oc row)));
    Bos_setup.R.ok ()
  in
  let _ = Bos.OS.File.with_oc fname f () in
  ()
;;

(* Stdio.Out_channel.with_file ~append:false ~fail_if_exists:false ~perm:777 fname ~f *)

let () =
  let dir_name = "/nfs/nomster/data/arul/data/artifacts/opt125m/fp32/layers/" in
  let files = Quantviz.Utils.dir_contents dir_name ~ext:"ot" in
  let ht = Hashtbl.create ~size:(List.length files) (module String) in
  List.iter ~f:(load_tensors ht) files;
  Hashtbl.iteri ht ~f:(fun ~key ~data -> write_csv key data);
  ()
;;
