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

let hist_columns = [ "layer_name"; "bin_start"; "bin_end"; "count"; "type_" ]
let calib_columns = [ "layer_name"; "type_"; "amax_type"; "amax_value" ]

let write_row oc row =
  Stdio.Out_channel.output_string oc row;
  Stdio.Out_channel.output_char oc '\n'
;;

let write_header oc columns =
  let header = String.concat ~sep:"," columns in
  write_row oc header
;;

let write_histogram csv_file layer_name (ttype, t) =
  let module H = Histogram in
  let f oc _ =
    write_header oc hist_columns;
    let t = Tensor.abs_ t in
    let tensor_max = find_max t in
    let x_max = Scalar.f tensor_max in
    let h_calib = H.make_t ~num_bins:2048 ~x_max in
    let counts = H.collect h_calib t in
    let percentile = 99 in
    let amax_perc =
      H.amax_percentile
        h_calib
        ~hist:counts
        ~numel:(numel t)
        ~percentile:(Float.of_int percentile)
    in
    let bins = Tensor.to_float1_exn h_calib.calib_bin_edges in
    let counts = Tensor.to_float1_exn counts in
    Array.iteri counts ~f:(fun idx count ->
      let bin_start = Float.to_string bins.(idx) in
      let bin_end = Float.to_string bins.(idx + 1) in
      let count = Float.to_string count in
      let row = [ layer_name; bin_start; bin_end; count; ttype ] in
      let row = String.concat ~sep:"," row in
      write_row oc row);
    Bos_setup.R.ok
      [ ( layer_name
        , ttype
        , Int.to_string percentile ^ "_percentile"
        , Float.to_string amax_perc )
      ; layer_name, ttype, "tensor_max", Float.to_string tensor_max
      ]
  in
  Bos.OS.File.with_oc (csv_file (ttype ^ "_hist")) f ()
;;

let extract_stats = Fn.compose Bos_setup.R.get_ok Bos_setup.R.get_ok

let write_calib csv_file ((ttype, _), calib_stats) =
  let calib_stats = extract_stats calib_stats in
  let f oc cs =
    write_header oc calib_columns;
    List.iter cs ~f:(fun (ln, ttype, amax_type, amax_value) ->
      let row = String.concat ~sep:"," [ ln; ttype; amax_type; amax_value ] in
      write_row oc row);
    Bos_setup.R.ok ()
  in
  Bos.OS.File.with_oc (csv_file (ttype ^ "_calib")) f calib_stats
;;

let write_csv layer_name names_and_tensors =
  let data_dir = "data" in
  let fname = Option.value_exn (Result.ok (Fpath.of_string data_dir)) in
  let _ = Bos.OS.Dir.create fname in
  let csv_file name = Fpath.add_seg fname name |> Fpath.add_ext "csv" in
  (* write histogram *)
  let calib_stats = List.map names_and_tensors ~f:(write_histogram csv_file layer_name) in
  (* write calib stats *)
  let zipped_stats = List.zip_exn names_and_tensors calib_stats in
  let _ = List.(zipped_stats >>| write_calib csv_file) in
  ()
;;

let () =
  let dir_name = "/nfs/nomster/data/arul/data/artifacts/opt125m/fp32/layers/" in
  let files = Quantviz.Utils.dir_contents dir_name ~ext:"ot" in
  let ht = Hashtbl.create ~size:(List.length files) (module String) in
  List.iter ~f:(load_tensors ht) files;
  (* Layers with weight *)
  let ht = Hashtbl.filter ht ~f:(fun v -> Hashtbl.mem v.layer_variables "weight") in
  Hashtbl.iteri ht ~f:(fun ~key ~data ->
    let names_and_tensors =
      [ (* "outputs", Hashtbl.find_exn data.outputs "0"; *)
        "inputs", Hashtbl.find_exn data.inputs "0"
      ]
      (* @ Hashtbl.to_alist data.layer_variables *)
    in
    write_csv key names_and_tensors);
  ()
;;
