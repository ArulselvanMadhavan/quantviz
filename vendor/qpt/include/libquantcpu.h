#pragma once

#include<stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
  typedef torch::Tensor *tensor;
#define PROTECT(x) \
  try { \
    x \
  } catch (const exception& e) { \
    caml_failwith(strdup(e.what())); \
  }  
  void at_print(tensor);
#ifdef __cplusplus
}
#endif  
