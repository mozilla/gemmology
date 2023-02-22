#include "gemmology.h"

#include <cmath>
#include <cstring>
#include <iostream>

namespace {

void QuantizeRef(const float *input, int8_t *output, float quant_mult, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    float value = roundf(input[i] * quant_mult);
    value = std::max(-127.0f, value);
    value = std::min(127.0f, value);
    output[i] = static_cast<int8_t>(value);
  }
}

template <class I> bool IsOff(float from, I ref, I test) {
  if (ref == test) return false;
  if (ref - test > 1 && test - ref > 1) return true;
  float off_test = std::fabs(static_cast<float>(test) - from);
  float off_ref = std::fabs(static_cast<float>(ref) - from);
  // Allow 0.5 to round either way.
  if (off_test > 0.49 && off_test < 0.51 && off_ref > 0.49 && off_ref < 0.51) return false;
  return true;
}

bool Test(const float *input_unaligned, float quant_mult, std::size_t size) {
  bool success = true;
  float *input;
  posix_memalign((void**)&input, 32, size * sizeof(*input));
  std::memcpy(input, input_unaligned, sizeof(*input) * size);

  int8_t *ref;
  posix_memalign((void**)&ref, 32, size * sizeof(*ref));
  QuantizeRef(input, ref, quant_mult, size);

  int8_t *test;
  posix_memalign((void**)&test, 32, size * sizeof(*test));
  gemmology::Quantize(input, test, quant_mult, size);

  for (std::size_t i = 0; i < size; ++i) {
    if (IsOff(input[i] * quant_mult, ref[i], test[i])) {
      std::cerr << "Error at " << i << " from " << input[i] << '*' << quant_mult << '=' << (input[i]*quant_mult) << " ref = " << static_cast<int>(ref[i]) << " test = " << static_cast<int>(test[i]) << "\n";
      success = false;
    }
  }
  return success;
}

bool TestMany(std::size_t grow) {
  float input[33] = {
    0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f,
    14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f,
    26.f, 27.f, 28.f, 29.f, 30.f, 31.f, 32.f};
  float corners[33] = {
    -32769.f, -32768.f, -32767.f, -129.f, -128.f, -127.f, -1.f, 0.f, 1.f,
    126.f, 127.f, 128.f, 129.f, 32766.f, 32768.f, 32769.f, -1.9f, -1.5f, -1.1f,
    -1.f, -0.9f, -0.5f, -0.1f, 0.0f, 0.1f, 0.5f, 0.9f, 1.0f, 1.1f, 1.5f, 1.9f,
    16056.8f, 2.5f};
  for (std::size_t len = 0; len <= 33; len += grow) {
    if(!Test(input, 1.0f, len)) return false;
    if(!Test(input, 32.0f, len)) return false;
    if(!Test(corners, 1.0f, len)) return false;
    if(!Test(corners, -1.0f, len)) return false;
    if(!Test(corners, -0.49f, len)) return false;
  }
  return true;
}

} // namespace

int main() {
  if(!TestMany(1))
    return 1;
  if(!TestMany(16))
    return 1;
}
