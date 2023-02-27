open Torch
open Base
open Calibrator
open Cmdliner

(* constants *)
let num_bins = 2048

let load_tensors ht filename =
  let contents = Serialize.load_all ~filename in
  let layer_info = Quantviz.Utils.layer_name_and_mem filename in
  let layer_name = List.hd_exn layer_info in
  let info_type = List.last_exn layer_info in
  Hashtbl.update ht layer_name ~f:Layercontents.(update info_type contents)
;;

let hist_columns = [ "layer_name"; "bin_start"; "bin_end"; "count"; "type_" ]
let mantissa_bits = [| 2; 3; 4; 5 |]

let fp_format m e =
  let m = Int.to_string m in
  let e = Int.to_string e in
  "M" ^ m ^ "E" ^ e
;;

let get_device device_id =
  if Cuda.is_available () && device_id > 0 then Device.Cuda device_id else Device.Cpu
;;

let get_channel_dim channel_dim = if channel_dim > 0 then Some channel_dim else None

let dump_hist_to_file percentiles oc layer_name ttype x_max t =
  let module H = Histogram in
  (* Write hist counts to file *)
  let x_max = Tensor.to_float0_exn x_max in
  let h_calib = H.make_t ~num_bins ~x_max in
  let counts = H.collect h_calib t in
  (* xmax_percentile *)
  let amax_perc =
    Array.map percentiles ~f:(fun percentile ->
      H.amax_percentile h_calib ~hist:counts ~numel:(Tensor_utils.numel t) ~percentile)
  in
  let bins = Tensor.to_float1_exn h_calib.calib_bin_edges in
  let counts = Tensor.to_float1_exn counts in
  Array.iteri counts ~f:(fun idx count ->
    let bin_start = Float.to_string bins.(idx) in
    let bin_end = Float.to_string bins.(idx + 1) in
    let count = Float.to_string count in
    let row = [ layer_name; bin_start; bin_end; count; ttype ] in
    let row = String.concat ~sep:"," row in
    Csv.write_row oc row);
  amax_perc
;;

let write_histogram channel_dim device_id percentiles oc layer_name (ttype, t) =
  let module H = Histogram in
  let calib_row name maxval mse sqnr m e =
    [ layer_name
    ; ttype
    ; name
    ; Tensor.minimum maxval |> Tensor.to_float0_exn |> Float.to_string
    ; Tensor.maximum maxval |> Tensor.to_float0_exn |> Float.to_string
    ; Float.to_string mse
    ; Float.to_string sqnr
    ; fp_format m e
    ; Int.to_string (Tensor_utils.numel t * 4)
    ]
  in
  let device = get_device device_id in
  let t = Tensor.to_ t ~device in
  let t = Tensor.abs_ t in
  let channel_dim = get_channel_dim channel_dim in
  let x_max = Tensor.maximum t in
  let amax_perc = dump_hist_to_file percentiles oc layer_name ttype x_max t in
  (* Mse *)
  let mse_results =
    Array.map mantissa_bits ~f:(fun mb ->
      Mse.amax_mse t ?channel_dim ~num_mantissa_bits:mb)
    |> Array.to_list
  in
  let i = ref 0 in
  let meandims = List.init (List.length (Tensor.shape t)) ~f:Fn.id in
  (* build calib rows *)
  let calc_mse_row mb exp (name, maxval) =
    let xfp = Mse.quantize_to_fp8 ?channel_dim t maxval ~num_mantissa_bits:mb in
    let mse = Mse.calc_mse t xfp meandims in
    let sqnr = Mse.calc_sqnr t xfp meandims |> Tensor.to_float0_exn in
    let mse = Tensor.to_float0_exn mse in
    calib_row name maxval mse sqnr mb exp
  in
  let expand_to_channel_dim maxval =
    Option.fold channel_dim ~init:maxval ~f:(fun maxval cdim ->
      let num_channels = Array.get (Tensor.shape t |> Array.of_list) cdim in
      Tensor.repeat maxval ~repeats:[ num_channels ])
  in
  let mse_result_to_row maxval =
    let mb = mantissa_bits.(!i) in
    let exp = 7 - mb in
    Stdio.printf "MSE result:%d|%d\n" mb exp;
    Stdio.Out_channel.flush Stdio.stdout;
    let mses =
      Array.mapi percentiles ~f:(fun i percentile ->
        let amax_perc = amax_perc.(i) |> Tensor.of_float0 ~device in
        Float.to_string percentile ^ "_percentile", expand_to_channel_dim amax_perc)
    in
    let mses =
      Array.to_list mses
      @ [ "tensor_max", expand_to_channel_dim x_max; fp_format mb exp, maxval ]
    in
    let mses = List.map mses ~f:(calc_mse_row mb exp) in
    i := !i + 1;
    (* To avoid GPU mem overflow when a long list of percentiles are computed *)
    Caml.Gc.full_major ();
    mses
  in
  List.(mse_results >>= mse_result_to_row)
;;

let write_csv
  hist_oc
  calib_oc
  device_id
  channel_dim
  percentiles
  layer_name
  names_and_tensors
  =
  (* write histogram *)
  Stdio.printf "Processing: %s\n" layer_name;
  Stdio.Out_channel.flush Stdio.stdout;
  let calib_stats =
    List.map
      names_and_tensors
      ~f:(write_histogram channel_dim device_id percentiles hist_oc layer_name)
  in
  (* write calib stats *)
  let zipped_stats = List.zip_exn names_and_tensors calib_stats in
  let _ = List.(zipped_stats >>| Csv.write_calib calib_oc) in
  ()
;;

let filter_float_tensors (_, t) =
  match Tensor.kind t with
  | T Float | T Double | T Half -> true
  | _ -> false
;;

let filter_by_info_type info_type filename =
  let layer_info = Quantviz.Utils.layer_name_and_mem filename in
  let layer_name = List.hd_exn layer_info in
  let it = List.last_exn layer_info in
  if String.equal info_type it
  then
    (* Stdio.printf "%s\n" filename; *)
    (* Stdio.Out_channel.flush Stdio.stdout; *)
    Some layer_name
  else None
;;

let info_type_to_tensors info_type (lc : Layercontents.t) =
  match info_type with
  | "inputs" -> [ info_type, Hashtbl.find_exn lc.inputs "0" ]
  | "output" -> [ info_type, Hashtbl.find_exn lc.outputs "0" ]
  | "layer_variables" -> [ "weight", Hashtbl.find_exn lc.layer_variables "weight" ]
  | _ -> []
;;

let filter_layers_by_name fname =
  (* Early stop *)
  let rec f = function
    | substring :: _ when String.is_substring fname ~substring -> false
    | _ :: xs -> f xs
    | [] -> true
  in
  f [ "layer_norm"; "activation_fn"; "embed" ]
;;

let file_stats dir_name =
  Stdio.printf "Is Cuda_avail:%b\n" (Cuda.is_available ());
  Stdio.Out_channel.flush Stdio.stdout;
  (* Select and filter files *)
  let files = Quantviz.Utils.dir_contents dir_name ~ext:"ot" in
  let files = List.rev files in
  let files = List.filter files ~f:filter_layers_by_name in
  let ht = Hashtbl.create ~size:(List.length files) (module String) in
  List.iter ~f:(load_tensors ht) files;
  let ht = Hashtbl.filter ht ~f:(fun v -> Hashtbl.mem v.layer_variables "weight") in
  files, ht
;;

let get_names_and_tensors info_type data =
  let names_and_tensors = info_type_to_tensors info_type data in
  List.filter names_and_tensors ~f:filter_float_tensors
;;

let csv_file name =
  let data_dir = "data" in
  let fname = Option.value_exn (Result.ok (Fpath.of_string data_dir)) in
  let _ = Bos.OS.Dir.create fname in
  Fpath.add_seg fname name |> Fpath.add_ext "csv"
;;

let run_vsq_sim device_id channel_dim dir_name info_type vsizes tensor_bits scale_bits =
  let files, ht = file_stats dir_name in
  let device = get_device device_id in
  let channel_dim = get_channel_dim channel_dim in
  let process_tensors oc layer_name =
    Stdio.printf "Layer:%s\n" layer_name;
    Stdio.Out_channel.flush Stdio.stdout;
    Option.fold (Hashtbl.find ht layer_name) ~init:() ~f:(fun _ data ->
      let named_tensors = get_names_and_tensors info_type data in
      Vsq.(
        quantize ?channel_dim device named_tensors ~vsizes ~tensor_bits ~scale_bits
        |> build_rows layer_name
        |> dump_rows oc);
      Caml.Gc.full_major ())
  in
  let f oc _ =
    Csv.write_header oc Csv.calib_columns;
    Bos_setup.R.return
      List.(
        filter_map files ~f:(filter_by_info_type info_type)
        |> iter ~f:(process_tensors oc))
  in
  Bos.OS.File.with_oc (csv_file ("vsq_" ^ info_type ^ "_calib")) f ()
  |> Bos_setup.R.get_ok
  |> Bos_setup.R.get_ok
;;

let run_fp8_sim device_id channel_dim dir_name info_type percentiles =
  let files, ht = file_stats dir_name in
  let hist_writer hist_oc calib_oc device_id =
    let process_tensors layer_name =
      Option.fold (Hashtbl.find ht layer_name) ~init:() ~f:(fun _ data ->
        let names_and_tensors = get_names_and_tensors info_type data in
        write_csv
          hist_oc
          calib_oc
          device_id
          channel_dim
          percentiles
          layer_name
          names_and_tensors)
    in
    List.(filter_map files ~f:(filter_by_info_type info_type) |> iter ~f:process_tensors)
  in
  (* Create file handlers *)
  let open_files handler =
    let write_to_files hist_oc calib_oc _ =
      Csv.write_header hist_oc hist_columns;
      Csv.write_header calib_oc Csv.calib_columns;
      Bos_setup.R.ok (handler hist_oc calib_oc device_id)
    in
    let open_calib_file hist_oc _ =
      let _ =
        Bos.OS.File.with_oc (csv_file (info_type ^ "_calib")) (write_to_files hist_oc) ()
      in
      Bos_setup.R.ok ()
    in
    let _ = Bos.OS.File.with_oc (csv_file (info_type ^ "_hist")) open_calib_file () in
    ()
  in
  open_files hist_writer
;;

let help_sections =
  [ `S Manpage.s_common_options
  ; `P "These options are common to all commands."
  ; `S "MORE HELP"
  ; `P "Use `$(mname) $(i,COMMAND) --help' for help on a single command."
  ; `Noblank
  ; `S Manpage.s_bugs
  ; `P "Check bug reports at https://github.com/ArulselvanMadhavan/quantviz/issues."
  ]
;;

let dir_arg =
  let doc = "Directory containing .ot files" in
  Arg.(required & pos 0 (some string) None & info [] ~docv:"DIRECTORY" ~doc)
;;

let device_arg =
  let doc = "Torch device to run on; Defaults to CPU, if not provided" in
  Arg.(value & opt int (-1) & info [ "d"; "cuda-device-id" ] ~doc)
;;

let channel_dim_arg =
  let doc =
    "Channel dim to use for per-channel quantization. If not provided, per-tensor \
     quantization will be used"
  in
  Arg.(value & opt int (-1) & info [ "c"; "channel-dim" ] ~doc)
;;

let info_type_arg =
  let doc = "inputs|outputs|layer_variables" in
  Arg.(
    required
    & pos 1 (some string) None
    & info [] ~docv:"TENSOR_TYPE:inputs|layer_variables|outputs" ~doc)
;;

let percentile_arg =
  let doc = "percentiles" in
  Arg.(
    required
    & pos 2 (some (array float)) None
    & info [] ~docv:"percentile list:[99.0,99.9,99.99]" ~doc)
;;

let vector_size_arg =
  let doc = "vector size" in
  Arg.(
    required & pos 2 (some (list int)) None & info [] ~docv:"VECTOR_SIZE:16,64,128" ~doc)
;;

let tensor_bits_arg =
  let doc = "bits to use to quantize the tensor" in
  Arg.(required & pos 3 (some (list int)) None & info [] ~docv:"tensor_bits:4,8" ~doc)
;;

let scale_bits_arg =
  let doc = "bits to use to quantize the scale tensor" in
  Arg.(required & pos 4 (some (list int)) None & info [] ~docv:"tensor_bits:6, 8,10" ~doc)
;;

let fp8_cmd =
  let doc = "Generate FP8 quantization errors" in
  let man =
    [ `S Manpage.s_description
    ; `P
        "Read the input directory recursively for .ot files and generate fp8 simulation \
         errors"
    ]
  in
  let info = Cmd.info "fp8" ~doc ~man in
  Cmd.v
    info
    Term.(
      const run_fp8_sim
      $ device_arg
      $ channel_dim_arg
      $ dir_arg
      $ info_type_arg
      $ percentile_arg)
;;

let vsquant_cmd =
  let doc = "Generate VS-Quant results" in
  let man =
    [ `S Manpage.s_description
    ; `P
        "Read the input directory recursively for .ot files and generate vsquant \
         simulation errors"
    ]
  in
  let info = Cmd.info "vsq" ~doc ~man in
  Cmd.v
    info
    Term.(
      const run_vsq_sim
      $ device_arg
      $ channel_dim_arg
      $ dir_arg
      $ info_type_arg
      $ vector_size_arg
      $ tensor_bits_arg
      $ scale_bits_arg)
;;

let main_cmd =
  let doc = "Generate FP8 simulation evaluation" in
  let sdocs = Manpage.s_common_options in
  let info = Cmd.info "quantviz" ~version:"dev" ~doc ~sdocs ~man:help_sections in
  let default = Term.(ret (const (`Help (`Pager, None)))) in
  Cmd.group info ~default [ fp8_cmd; vsquant_cmd ]
;;

let () = Stdlib.exit (Cmd.eval main_cmd)
