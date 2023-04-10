include module type of struct include Torch_core.Wrapper.Tensor end

module C : module type of Qtorch_bindings.C(Qtorch_generated) with type t := Torch_core.Wrapper_generated.C.Tensor.t
  
val tensor_to_int : t -> int
