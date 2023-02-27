open Base

let calib_columns =
  [ "layer_name"
  ; "type_"
  ; "amax_type"
  ; "amax_start"
  ; "amax_end"
  ; "mse"
  ; "sqnr"
  ; "format"
  ; "scale_size"
  ]
;;

let write_row oc row =
  Stdio.Out_channel.output_string oc row;
  Stdio.Out_channel.output_char oc '\n'
;;

let write_header oc columns =
  let header = String.concat ~sep:"," columns in
  write_row oc header
;;

let write_calib calib_oc (_, calib_stats) =
  List.iter calib_stats ~f:(fun xs ->
    let row = String.concat ~sep:"," xs in
    write_row calib_oc row)
;;
