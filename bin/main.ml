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
let calib_columns = [ "layer_name"; "type_"; "amax_type"; "amax_value"; "mse"; "format" ]

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
  let calib_row name maxval mse m e =
    layer_name, ttype, name, Float.to_string maxval, Float.to_string mse, fp_format m e
  in
  (* Histogram *)
  let t = Tensor.to_ t ~device:(Device.Cuda 0) in
  let t = Tensor.abs_ t in
  let x_max = find_max t in
  let h_calib = H.make_t ~num_bins ~x_max in
  let counts = H.collect h_calib t in
  (* xmax_percentile *)
  let amax_perc =
    H.amax_percentile
      h_calib
      ~hist:counts
      ~numel:(numel t)
      ~percentile:(Float.of_int percentile)
  in
  (* Mse *)
  let mse_results =
    Array.map mantissa_bits ~f:(fun mb -> Mse.amax_mse t ~x_max ~num_mantissa_bits:mb)
    |> Array.to_list
  in
  (* Write hist counts to file *)
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
  let meandims = List.init (List.length (Tensor.shape t)) ~f:Fn.id in
  (* build calib rows *)
  let calc_mse_row mb exp (name, maxval) =
    let maxval_t = Tensor.of_float0 ~device:(Tensor.device t) maxval in
    let xfp = Mse.quantize_to_fp8 t maxval_t ~num_mantissa_bits:mb in
    let mse = Mse.calc_mse t xfp meandims in
    let mse = Tensor.to_float0_exn mse in
    calib_row name maxval mse mb exp
  in
  let mse_result_to_row mse_res =
    let mb = mantissa_bits.(!i) in
    let exp = 7 - mb in
    let need_mses =
      [ Int.to_string percentile ^ "_percentile", amax_perc; "tensor_max", x_max ]
    in
    let mses = List.map need_mses ~f:(calc_mse_row mb exp) in
    let mse_pos, mse_val = mse_res in
    let mse_pos = Array.zip_exn mse_pos mse_val |> Array.to_list in
    i := !i + 1;
    mses
    @ List.map mse_pos ~f:(fun (maxval, mse) -> calib_row "min_mse" maxval mse mb exp)
  in
  List.(mse_results >>= mse_result_to_row)
;;

let write_calib calib_oc (_, calib_stats) =
  List.iter calib_stats ~f:(fun (ln, ttype, amax_type, amax_value, mse, format) ->
    let row = String.concat ~sep:"," [ ln; ttype; amax_type; amax_value; mse; format ] in
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

let info_type = "inputs"

let filter_by_info_type filename =
  let layer_info = Quantviz.Utils.layer_name_and_mem filename in
  let layer_name = List.hd_exn layer_info in
  let it = List.last_exn layer_info in
  if String.equal info_type it then Some layer_name else None
;;

let info_type_to_tensors (lc : Layercontents.t) =
  match info_type with
  | "inputs" -> [ info_type, Hashtbl.find_exn lc.inputs "0" ]
  | "output" -> [ info_type, Hashtbl.find_exn lc.outputs "0" ]
  | "layer_variables" -> [ "weight", Hashtbl.find_exn lc.layer_variables "weight" ]
  | _ -> []
;;

let () =
  let dir_name = "/nfs/nomster/data/arul/data/artifacts/opt125m/fp32/layers/" in
  let files = Quantviz.Utils.dir_contents dir_name ~ext:"ot" in
  let ht = Hashtbl.create ~size:(List.length files) (module String) in
  List.iter ~f:(load_tensors ht) files;
  (* Layers with weight *)
  let ht = Hashtbl.filter ht ~f:(fun v -> Hashtbl.mem v.layer_variables "weight") in
  let hist_writer hist_oc calib_oc =
    let process_tensors layer_name =
      Option.fold (Hashtbl.find ht layer_name) ~init:() ~f:(fun _ data ->
        let names_and_tensors = info_type_to_tensors data in
        let names_and_tensors = List.filter names_and_tensors ~f:filter_float_tensors in
        write_csv hist_oc calib_oc layer_name names_and_tensors)
    in
    (* To maintain order iter through files in the order *)
    List.(filter_map files ~f:filter_by_info_type |> iter ~f:process_tensors)
  in
  let data_dir = "data" in
  let fname = Option.value_exn (Result.ok (Fpath.of_string data_dir)) in
  let _ = Bos.OS.Dir.create fname in
  let csv_file name = Fpath.add_seg fname name |> Fpath.add_ext "csv" in
  let _ =
    Bos.OS.File.with_oc
      (csv_file (info_type ^ "_hist"))
      (fun hist_oc _ ->
        write_header hist_oc hist_columns;
        let _ =
          Bos.OS.File.with_oc
            (csv_file (info_type ^ "_calib"))
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
