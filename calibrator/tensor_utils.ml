open Torch
open Base

let numel t = List.fold (Tensor.shape t) ~init:1 ~f:Int.( * )
