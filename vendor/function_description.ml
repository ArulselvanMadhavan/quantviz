open Ctypes

module Types = Types_generated
  
module Functions (F : Ctypes.FOREIGN) = struct  

  open F
      
  let print_int = foreign "print_int" (int @-> returning void)
      
end
