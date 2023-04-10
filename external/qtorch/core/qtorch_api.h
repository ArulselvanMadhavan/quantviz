#ifndef __QTORCH_API_H__
#define __QTORCH_API_H__
#include<stdint.h>

#ifdef __cplusplus
extern "C" {
typedef torch::Tensor *tensor;
#define PROTECT(x) \
  try { \
    x \
  } catch (const exception& e) { \
    caml_failwith(strdup(e.what())); \
  }
#else
typedef void *tensor;  
#endif

tensor tensor_to_int();
#ifdef __cplusplus
};
#endif
#endif  
