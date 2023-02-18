(* open Torch *)

exception UnrecognizedStateDictType of string

let dir_contents dir =
  let rec loop result = function
    | f :: fs when Sys.is_directory f ->
      Sys.readdir f
      |> Array.to_list
      |> List.map (Filename.concat f)
      |> List.append fs
      |> loop result
    | f :: fs when String.ends_with ~suffix:".pt" f -> loop (f :: result) fs
    | _f :: fs -> loop result fs
    | [] -> result
  in
  loop [] [ dir ]
;;

let _print_type obj =
  let typ = Py.Type.get obj in
  let str = Py.Type.name typ in
  Stdio.printf "Type:%s\n" str
;;

let _print_shape shp =
  let typ = Py.Type.get shp in
  match typ with
  | Py.Type.Tuple ->
    let shp_str =
      Py.Tuple.fold_left (fun acc obj -> acc ^ "," ^ Py.Int.to_string obj) "" shp
    in
    Stdio.printf "Shape:%s\n" shp_str
  | _ -> Stdio.printf "!!!Unexpected type:%s\n" (Py.Type.name typ)
;;

let save_npz dst kwargs =
  let np = Py.import "numpy" in
  Py.Module.get_function_with_keywords np "savez" [| Py.String.of_string dst |] kwargs
;;

let enumerate_list args = List.mapi (fun i l -> Int.to_string i, l) args

let pt_to_npy src dst =
  let open Pyops in
  let tch = Py.import "torch" in
  let load_arg = Py.String.of_string src in
  let tensor = tch.&("load") [| load_arg |] in
  (match Py.Type.get tensor with
   | Py.Type.Tuple ->
     let args = enumerate_list (Py.Tuple.to_list tensor) in
     let _ = save_npz dst args in
     ()
   | Py.Type.Dict ->
     let items = Py.Dict.items tensor in
     let items =
       Py.List.to_list_map
         (fun tup ->
           let key = Py.Tuple.get_item tup 0 in
           let key = Py.String.to_string key in
           key, Py.Tuple.get_item tup 1)
         items
     in
     let _ = save_npz dst items in
     ()
   | Py.Type.Iter ->
     let args = [ Int.to_string 0, tensor ] in
     let _ = save_npz dst args in
     ()
   | _ -> raise (UnrecognizedStateDictType src));
  ()
;;

let remove_file_ext files =
  let open Base in
  let src_files = List.filter_map files ~f:Result.ok in
  let dst_files = List.map src_files ~f:(fun fl -> Fpath.rem_ext fl) in
  List.zip_exn src_files dst_files
;;

let () =
  let open Base in
  Py.initialize ();
  let dir_name = "/nfs/nomster/data/arul/data/artifacts/opt125m/fp32/layers/" in
  let files = dir_contents dir_name in
  let files = List.map files ~f:(fun fl -> Fpath.of_string fl) in
  let files = remove_file_ext files in
  List.iter files ~f:(fun (src, dst) ->
    pt_to_npy (Fpath.to_string src) (Fpath.to_string dst));
  List.iter ~f:(fun (f, _) -> Stdio.printf "%s\n" (Fpath.to_string f)) files
;;
