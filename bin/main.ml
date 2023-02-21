open Torch
open Base
open Calibrator

(* constants *)
let percentile = 99
let num_bins = 2048

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

let mantissa_bits = [| 5; 4 |]

let fp_format m e =
  let m = Int.to_string m in
  let e = Int.to_string e in
  "M" ^ m ^ "E" ^ e
;;

let write_histogram oc layer_name (ttype, t) =
  let module H = Histogram in
  Stdio.printf "%s\n" layer_name;
  Stdio.Out_channel.flush Stdio.stdout;
  let t = Tensor.abs_ t in
  let x_max = find_max t in
  let h_calib = H.make_t ~num_bins ~x_max in
  let counts = H.collect h_calib t in
  let amax_perc =
    H.amax_percentile
      h_calib
      ~hist:counts
      ~numel:(numel t)
      ~percentile:(Float.of_int percentile)
  in
  (* let t = Tensor.to_ t ~device:(Device.Cuda 0) in *)
  let mse_results =
    Array.map mantissa_bits ~f:(fun mb -> Mse.amax_mse t ~x_max ~num_mantissa_bits:mb)
    |> Array.to_list
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
  let i = ref 0 in
  let mse_rows =
    List.(
      mse_results
      >>= fun mse_res ->
      let mb = mantissa_bits.(!i) in
      let exp = 7 - mb in
      let mse_pos, _mse_val = mse_res in
      let mse_pos = Array.to_list mse_pos in
      i := !i + 1;
      List.map mse_pos ~f:(fun pos ->
        layer_name, ttype, fp_format mb exp ^ "_min_mse", Float.to_string pos))
  in
  [ layer_name, ttype, Int.to_string percentile ^ "_percentile", Float.to_string amax_perc
  ; layer_name, ttype, "tensor_max", Float.to_string x_max
  ]
  @ mse_rows
;;

let write_calib calib_oc (_, calib_stats) =
  List.iter calib_stats ~f:(fun (ln, ttype, amax_type, amax_value) ->
    let row = String.concat ~sep:"," [ ln; ttype; amax_type; amax_value ] in
    write_row calib_oc row)
;;

let write_csv hist_oc calib_oc layer_name names_and_tensors =
  (* write histogram *)
  let calib_stats = List.map names_and_tensors ~f:(write_histogram hist_oc layer_name) in
  (* write calib stats *)
  let zipped_stats = List.zip_exn names_and_tensors calib_stats in
  let _ = List.(zipped_stats >>| write_calib calib_oc) in
  ()
;;

let filter_float_tensors (_, t) =
  match Tensor.kind t with
  | T Float | T Double | T Half -> true
  | _ -> false
;;

let () =
  let dir_name = "/nfs/nomster/data/arul/data/artifacts/opt125m/fp32/layers/" in
  let files = Quantviz.Utils.dir_contents dir_name ~ext:"ot" in
  let ht = Hashtbl.create ~size:(List.length files) (module String) in
  List.iter ~f:(load_tensors ht) files;
  (* Layers with weight *)
  let ht = Hashtbl.filter ht ~f:(fun v -> Hashtbl.mem v.layer_variables "weight") in
  (* FIXME *)
  let ttype = "inputs" in
  let hist_writer hist_oc calib_oc =
    List.iter files ~f:(fun filename ->
      let layer_info = Quantviz.Utils.layer_name_and_mem filename in
      let layer_name = List.hd_exn layer_info in
      Option.fold (Hashtbl.find ht layer_name) ~init:() ~f:(fun _ data ->
        let names_and_tensors =
          [ (* "outputs", Hashtbl.find_exn data.outputs "0"; *)
            "inputs", Hashtbl.find_exn data.inputs "0"
          ]
          (* @ Hashtbl.to_alist data.layer_variables *)
        in
        let names_and_tensors = List.filter names_and_tensors ~f:filter_float_tensors in
        write_csv hist_oc calib_oc layer_name names_and_tensors))
  in
  let data_dir = "data" in
  let fname = Option.value_exn (Result.ok (Fpath.of_string data_dir)) in
  let _ = Bos.OS.Dir.create fname in
  let csv_file name = Fpath.add_seg fname name |> Fpath.add_ext "csv" in
  let _ =
    Bos.OS.File.with_oc
      (csv_file (ttype ^ "_hist"))
      (fun hist_oc _ ->
        write_header hist_oc hist_columns;
        let _ =
          Bos.OS.File.with_oc
            (csv_file (ttype ^ "_calib"))
            (fun calib_oc _ ->
              write_header calib_oc calib_columns;
              Bos_setup.R.ok (hist_writer hist_oc calib_oc))
            ()
        in
        Bos_setup.R.ok ())
      ()
  in
  ()
;;
