#include "gemmology.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

namespace {
template <typename Type>
void TransposeRef(const Type* input, Type* output, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      output[rows * c + r] = input[cols * r + c];
    }
  }
}

bool TestTranspose16() {
  const unsigned N = 8;
  int size = N * N;
  int16_t* input;
  posix_memalign((void**)&input, 32, size * sizeof(*input));
  std::iota(input, input + size, static_cast<int16_t>(0));

  int16_t* ref;
  posix_memalign((void**)&ref, 32, size * sizeof(*ref));
  TransposeRef(input, ref, N, N);

  // Overwrite input.
  xsimd::batch<int8_t, xsimd::sse2> *t = (xsimd::batch<int8_t, xsimd::sse2> *)input;
  gemmology::Transpose16InLane(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);

  for (int i = 0; i < size; ++i) {
  	if(ref[i] != input[i]) {
      std::cerr << "16-bit transpose failure at: " << i << ": " << ref[i] << " != " << input[i] << "\n";
      return false;
    }
  }

  free(input);
  free(ref);
  return true;
}

}

int main() {
  if(!TestTranspose16())
    return 1;
  return 0;
}
