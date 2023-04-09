module type Tensor = module type of Torch_core.Wrapper.Tensor
module type C = module type of Qtorch_bindings.C(Qtorch_generated)
    
module type Qtensor = sig
  type t
  val tensor_to_int : t -> int
end

module type Builder = sig
  module Make(T : Tensor) : Qtensor with type t := T.t
end

(* module Qtensor(T : Tensor) = struct *)
(*   include T *)

(*   module C = Qtorch_bindings.C(Qtorch_generated) *)
(*   let tensor_to_int t = *)
(*     C.Tensor.tensor_to_int t *)
(* end *)

