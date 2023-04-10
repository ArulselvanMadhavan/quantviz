#include<torch/csrc/autograd/engine.h>
#include<torch/torch.h>
#include<ATen/autocast_mode.h>
#include<torch/script.h>
#include<torch/csrc/jit/mobile/import_data.h>
#include<stdexcept>
#include<vector>
#include<caml/fail.h>
#undef invalid_argument
#include "qtorch_api.h"

using namespace std;

int tensor_to_int(tensor t) {
  PROTECT(
    return 42;
    // return new torch::Tensor();
  )
  return -1;
}
