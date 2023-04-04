#include "libquantcpu.h"
#include <torch/torch.h>
#include <iostream>

void at_print(tensor t) {
  PROTECT(
    torch::Tensor *tensor = (torch::Tensor*)t;
    cout << *tensor << endl;
  )
}
