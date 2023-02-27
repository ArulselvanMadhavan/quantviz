let dir_contents dir ~ext =
  let rec loop result = function
    | f :: fs when Sys.is_directory f ->
      Sys.readdir f
      |> Array.to_list
      |> List.map (Filename.concat f)
      |> List.append fs
      |> loop result
    | f :: fs when String.ends_with ~suffix:ext f -> loop (f :: result) fs
    | _f :: fs -> loop result fs
    | [] -> result
  in
  loop [] [ dir ]
;;

let remove_file_ext files =
  let open Base in
  let src_files = List.filter_map files ~f:Result.ok in
  let dst_files = List.map src_files ~f:(fun fl -> Fpath.rem_ext fl) in
  List.zip_exn src_files dst_files
;;

let rename_ext files ~ext =
  let open Base in
  let src_files = List.filter_map files ~f:Result.ok in
  let dst_files = List.map src_files ~f:(fun fl -> Fpath.(fl -+ ext)) in
  List.zip_exn src_files dst_files
;;

let layer_name_and_mem fname =
  let fname = Fpath.of_string fname in
  let segs = Fpath.segs (Result.get_ok fname) in
  let open Base in
  let segs = Array.of_list segs in
  let segs_ht = Hashtbl.create ~size:(Array.length segs) (module String) in
  Array.iteri segs ~f:(fun i n -> Hashtbl.update segs_ht n ~f:(fun _ -> i));
  let layers_idx = Hashtbl.find_exn segs_ht "layers" in
  let lname_idx = layers_idx + 1 in
  let layer_name = segs.(lname_idx) in
  let info_type = segs.(lname_idx + 1) in
  layer_name, info_type
;;

let model_name dir_name =
  let fname = Fpath.of_string dir_name in
  let segs = Fpath.segs (Result.get_ok fname) in
  let open Base in
  let segs = Array.of_list segs in
  let segs_ht = Hashtbl.create ~size:(Array.length segs) (module String) in
  Array.iteri segs ~f:(fun i n -> Hashtbl.update segs_ht n ~f:(fun _ -> i));
  let artifacts_idx = Hashtbl.find_exn segs_ht "artifacts" in
  let model_name = segs.(artifacts_idx + 1) in
  model_name
;;
