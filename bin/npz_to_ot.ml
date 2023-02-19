open Torch
open Base

let npz_tensors ~filename ~f =
  let npz_file = Npy.Npz.open_in filename in
  let named_tensors =
    Npy.Npz.entries npz_file
    |> List.map ~f:(fun tensor_name -> f tensor_name (Npy.Npz.read npz_file tensor_name))
  in
  Npy.Npz.close_in npz_file;
  named_tensors
;;

let npz_to_pytorch npz_src pytorch_dst =
  let named_tensors =
    npz_tensors ~filename:npz_src ~f:(fun tensor_name packed_tensor ->
      match packed_tensor with
      | Npy.P tensor ->
        (match Bigarray.Genarray.layout tensor with
         | Bigarray.C_layout -> tensor_name, Tensor.of_bigarray tensor
         | Bigarray.Fortran_layout -> failwith "fortran layout is not supported"))
  in
  Serialize.save_multi ~named_tensors ~filename:pytorch_dst
;;

let () =
  let dir_name = "/nfs/nomster/data/arul/data/artifacts/opt125m/fp32/layers/" in
  let files = Quantviz.Utils.dir_contents dir_name ~ext:"npz" in
  let files = List.map files ~f:(fun fl -> Fpath.of_string fl) in
  let files = Quantviz.Utils.rename_ext files ~ext:"ot" in
  List.iter files ~f:(fun (src, dst) ->
    npz_to_pytorch (Fpath.to_string src) (Fpath.to_string dst));
  ()
;;
