(* include Qtensor_intf *)

    
(* module Make(T : Tensor) : Qtensor = struct   *)
(*   module C = Qtorch_bindings.C(Qtorch_generated) *)
(*   open! C.Tensor *)
      
(*   type nonrec t = t *)
    
(*   type t1 = T.t *)

(*   type t2 = t and t3 = t *)

(*   let tensor_to_int t = *)
(*     let result = C.Tensor.tensor_to_int t in *)
(*     result *)

(*     (\* C.Tensor.tensor_to_int t *\) *)
(* end *)

include Torch_core.Wrapper.Tensor

module type QC = module type of Qtorch_bindings.C(Qtorch_generated)
module C : QC with type t := Torch_core.Wrapper_generated.C.Tensor.t = Qtorch_bindings.C(Qtorch_generated)

let tensor_to_int t =
  C.tensor_to_int t
