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
  let segs = Base.List.drop_while segs ~f:(fun seg -> Base.String.(seg <> "layers")) in
  let segs = List.tl segs in
  Base.List.drop_last_exn segs
;;
