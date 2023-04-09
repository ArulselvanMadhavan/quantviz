open Ctypes

(* module QT = struct *)
(*   type t *)

(*   let t = Torch_core.Wrapper.Tensor.t *)
(* end *)

(* module type QT = sig *)
(*   module Tensor(T : ) *)
(* end *)
(* module type Tensor = module type of Torch_core.Wrapper.Tensor *)

(* module type QT = functor (F : Cstubs.FOREIGN) -> sig *)
(*   type t *)
    
(*   val tensor_to_int : (t -> int F.return) F.result *)
(* end *)

(* module C(F : Cstubs.FOREIGN) = struct *)
(*   open F *)

(*   module Tensor = struct *)
(*     type t = unit ptr *)

(*     let t = ptr void *)

(*     let tensor_to_int = foreign "tensor_to_int" (t @-> returning int) *)
(*   end *)
  
(* end *)

module type QB = functor (F : Cstubs.FOREIGN) -> sig
  type t = Torch_core.Wrapper_generated.C.Tensor.t

  val t : t typ

  val tensor_to_int : (t -> int F.return) F.result
end

module C : QB = functor (F : Cstubs.FOREIGN) -> struct
  open F

  type t = Torch_core.Wrapper_generated.C.Tensor.t

  let t = Torch_core.Wrapper_generated.C.Tensor.t

  let tensor_to_int = foreign "tensor_to_int" (t @-> returning int)
end

