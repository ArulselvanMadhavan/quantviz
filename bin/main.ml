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

let find_max t =
  let ndims = List.length (Tensor.shape t) in
  let x_max = Tensor.max_values t ~dim:(List.init ndims ~f:Fn.id) ~keepdim:false in
  Tensor.to_float0_exn x_max
;;

let hist_columns = [ "layer_name"; "bin_start"; "bin_end"; "count"; "type_"; "amax_perc" ]
let calib_columns = [ "layer_name"; "type_"; "amax_type"; "amax_value" ]

let write_row oc row =
  Stdio.Out_channel.output_string oc row;
  Stdio.Out_channel.output_char oc '\n'
;;

let write_header oc columns =
  let header = String.concat ~sep:"," columns in
  write_row oc header
;;

let write_csv layer_name (data : Layercontents.t) =
  let module H = Histogram in
  let data_dir = "data" in
  let fname = Option.value_exn (Result.ok (Fpath.of_string data_dir)) in
  (* let fname = Fpath.add_seg fname layer_name in *)
  let _ = Bos.OS.Dir.create fname in
  let csv_file name = Fpath.add_seg fname name |> Fpath.add_ext "csv" in
  let f oc _ =
    write_header oc hist_columns;
    let names_and_tensors =
      [ "outputs", Hashtbl.find_exn data.outputs "0"
      ; "inputs", Hashtbl.find_exn data.inputs "0"
      ]
      @ Hashtbl.to_alist data.layer_variables
    in
    (* ttype: inputs, outputs, keys(layer_variables) *)
    let calib_stats =
      List.map names_and_tensors ~f:(fun (ttype, t) ->
        let t = Tensor.abs_ t in
        let x_max = Scalar.f (find_max t) in
        let h_calib = H.make_t ~num_bins:2048 ~x_max in
        let counts = H.collect h_calib t in
        let amax_perc = H.amax_percentile h_calib ~hist:counts ~numel:(numel t) in
        let amax_perc = Float.to_string amax_perc in
        let bins = Tensor.to_float1_exn h_calib.calib_bin_edges in
        let counts = Tensor.to_float1_exn counts in
        Array.iteri counts ~f:(fun idx count ->
          let bin_start = Float.to_string bins.(idx) in
          let bin_end = Float.to_string bins.(idx + 1) in
          let count = Float.to_string count in
          let row = [ layer_name; bin_start; bin_end; count; ttype; amax_perc ] in
          let row = String.concat ~sep:"," row in
          write_row oc row);
        [ layer_name, ttype, "amax_percentile", amax_perc ])
    in
    Bos_setup.R.ok (List.concat calib_stats)
  in
  let calib_stats = Bos.OS.File.with_oc (csv_file "hist") f () in
  let calib_stats = Bos_setup.R.get_ok calib_stats |> Bos_setup.R.get_ok in
  let f oc cs =
    write_header oc calib_columns;
    List.iter cs ~f:(fun (ln, ttype, amax_type, amax_value) ->
      let row = String.concat ~sep:"," [ ln; ttype; amax_type; amax_value ] in
      write_row oc row);
    Bos_setup.R.ok ()
  in
  let _ = Bos.OS.File.with_oc (csv_file "calib") f calib_stats in
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
