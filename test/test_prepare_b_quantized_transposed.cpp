#include "gemmology.h"

#include <cmath>
#include <cstring>
#include <iostream>

namespace {

void PrepareBQuantizedTransposedRef(const int8_t* input, int8_t* output, int B_transposed_cols, int B_transposed_rows) {
  constexpr int vec_len = sizeof(xsimd::batch<int8_t>) / sizeof(int8_t);

  auto output_it = output;
  for (int r = 0; r < B_transposed_rows; r += 8)
    for (int c = 0; c < B_transposed_cols; c += vec_len)
      for (int ri = 0; ri < 8; ++ri)
        for (int ci = 0; ci < vec_len; ++ci)
          *output_it++ = input[(r + ri) * B_transposed_cols + c + ci];
}

bool Test(const int8_t * input, int B_rows, int B_cols) {
  bool success = true;

  int input_size = B_rows * B_cols;

  int8_t * output;
  posix_memalign((void**)&output, 64, input_size * sizeof(*output));
  gemmology::PrepareBQuantizedTransposed(input, output, B_rows, B_cols);

  int8_t * reference;
  posix_memalign((void**)&reference, 64, input_size * sizeof(*reference));
  PrepareBQuantizedTransposedRef(input, reference, B_rows, B_cols);

  for (int i = 0; i < input_size; ++i) {
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

bool TestMany(int B_rows, int B_cols) {
  int8_t * input;
  int input_size = B_rows * B_cols;
  posix_memalign((void**)&input, 64, input_size * sizeof(*input));

  std::generate(input, input + input_size, []() {
    static constexpr int divider = sizeof(xsimd::batch<int8_t>) / sizeof(int8_t);
    static int value = 0;
    return static_cast<int8_t>((value++) % divider);
  });

  bool res =  Test(input, B_rows, B_cols);
  free(input);
  return res;
}
}

int main() {
  if(!TestMany(64, 128))
    return 1;
  if(!TestMany(512, 512))
    return 1;
  return 0;
}
