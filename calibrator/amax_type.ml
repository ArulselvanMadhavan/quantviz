open Base

type t =
  | Tensor
  | Channel of int
  | Vector of int
[@@deriving sexp]

let from_channel_dim cdim = Option.fold cdim ~init:Tensor ~f:(fun _ dim -> Channel dim)

let to_channel_dim = function
  | Tensor -> None
  | Channel dim -> Some dim
  | _ -> None
;;
