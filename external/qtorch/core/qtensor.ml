include Qtensor_intf

    
module Make(T : Tensor) : Qtensor = struct  
  module C = Qtorch_bindings.C(Qtorch_generated)
  open! C.Tensor
      
  type nonrec t = T.t
  let tensor_to_int (t : T.t) =
    let result = C.Tensor.tensor_to_int t in
    result
    (* C.Tensor.tensor_to_int t *)
end

