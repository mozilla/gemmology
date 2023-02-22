#include "gemmology.h"

#include <cmath>
#include <cstring>
#include <iostream>

using namespace gemmology;

namespace {

void PrepareBTransposedRef(const float* input, int8_t* output, float quant_mult, int B_transposed_cols, int B_transposed_rows) {
  constexpr int vec_len = sizeof(xsimd::batch<int8_t>) / sizeof(int8_t);

  for (int i = 0; i < B_transposed_rows * B_transposed_cols / 8; i += vec_len)
    for (int j = 0; j < 8; ++j)
      for (int k = 0; k < vec_len; ++k) {
        int col = (i + k) % B_transposed_cols;
        int row = 8 * ((i + k) / B_transposed_cols) + j;
        *output++ = static_cast<int8_t>(input[row * B_transposed_cols + col] * quant_mult);
      }
}

bool Test(const float* input, int B_rows, int B_cols, float quant_mult) {
  bool success = true;

  int8_t *output;
  int input_size =  B_rows * B_cols;
  posix_memalign((void**)&output, 64, input_size * sizeof(*output));
  PrepareBTransposed(input, output, quant_mult, B_rows, B_cols);

  int8_t *reference;
  posix_memalign((void**)&reference, 64, input_size * sizeof(*reference));
  PrepareBTransposedRef(input, reference, quant_mult, B_rows, B_cols);

  for (std::size_t i = 0; i < B_rows * B_cols * sizeof(*output); ++i) {
    if (output[i] != reference[i]) {
      std::cerr << "Error at " << i << ", output = " << int(output[i]) << ", reference = " << int(reference[i]) << std::endl;
      success = false;
      break;
    }
  }

  free(output);
  free(reference);
  return success;
}

bool TestMany(int B_rows, int B_cols, float quant_mult) {
  float *input;
  int input_size = B_rows * B_cols;
  posix_memalign((void**)&input, 64, input_size * sizeof(*input));

  std::generate(input, input + input_size, []() {
    static constexpr int divider = sizeof(xsimd::batch<int8_t>) / sizeof(int8_t);
    static int value = 0;
    return static_cast<float>((value++) % divider);
  });

  bool res = Test(input, B_rows, B_cols, quant_mult);
  free(input);
  return res;
}
}

int main() {
  if(!TestMany(16, 128, 2.0f))
    return 1;
  return 0;
}

