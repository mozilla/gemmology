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

bool CompareAs(int8_t *output_old, uint8_t *output_new, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      int a = int(output_old[rows * c + r]);
      int b = int(output_new[rows * c + r]);
      if (a + 127 != b) {
        std::cerr << "compare failed\n";
        return false;
      }
    }
  }
  return true;
}

template <typename Type>
bool CompareEps(const Type *reference, const Type *actual, int size,
                Type epsilon) {
  for (int i = 0; i < size; ++i) {
    // Ratio to maximum value.
    float threshold = epsilon * std::max<float>(0.01f, std::fabs(reference[i]));
    if (std::fabs(reference[i] - actual[i]) >= threshold) {
      std::cerr << "mismatch: " << reference[i]  << " vs " << actual[i]<< "\n";
      return false;
    }
  }
  return true;
}

bool CompareMSE(const float *float_ref, const float *int_ref,
                const float *int_test, std::size_t size, float int_tolerance,
                float float_tolerance, float MSE_float_tolerance,
                float MSE_int_tolerance) {
  float int_sum = 0.0, float_sum = 0.0;
  for (std::size_t i = 0; i < size; ++i) {
    float int_diff = int_ref[i] - int_test[i];
    float float_diff = float_ref[i] - int_test[i];
    if (std::fabs(int_diff) > int_tolerance) {
      std::cerr << "Inaccurate compared to int reference at " << i << ' '
                << int_ref[i] << ' ' << int_test[i] << "\n";
      return false;
    }
    if (std::fabs(float_diff) > float_tolerance) {
      std::cerr << "Inaccurate compared to float reference at " << i << ' '
                << float_ref[i] << ' ' << int_test[i] << "\n";
      return false;
    }
    int_sum += int_diff * int_diff;
    float_sum += float_diff * float_diff;
  }
  if (std::fabs(sqrt(float_sum / size)) > MSE_float_tolerance) {
    std::cerr << "Float MSE = " << sqrt(float_sum / size) << "\n";
    return false;
  }
  if (std::fabs(sqrt(int_sum / size)) > MSE_int_tolerance) {
    std::cerr << "Int MSE = " << sqrt(int_sum / size) << "\n";
    return false;
  }
  return true;
}

template <typename Type>
void RearragementRef(const Type *input, Type *output, int simd, int unroll,
                     int rows, int cols) {
  for (int c = 0; c < cols; c += unroll) {
    for (int r = 0; r < rows; r += simd) {
      for (int i = 0; i < unroll; ++i)
        for (int j = 0; j < simd; ++j)
          output[simd * i + j] = input[cols * r + c + cols * j + i];

      output += unroll * simd;
    }
  }
}

template <
    typename TypeA, typename TypeB, typename TypeC, typename LambdaCallback,
    typename std::enable_if<
        (std::is_integral<TypeA>::value && std::is_integral<TypeB>::value) ||
        (std::is_floating_point<TypeA>::value &&
         std::is_floating_point<TypeB>::value)>::type * = nullptr>
void MultiplyRef(const TypeA *A, const TypeB *B, TypeC *C, int A_rows,
                 int width, int B_cols, LambdaCallback callback) {
  using IntermediateType =
      typename std::conditional<std::is_integral<TypeA>::value, int32_t,
                                double>::type;

  for (int r = 0; r < A_rows; ++r) {
    for (int c = 0; c < B_cols; ++c) {
      IntermediateType sum = 0;
      for (int k = 0; k < width; ++k) {
        sum += IntermediateType(A[r * width + k]) *
               IntermediateType(B[k * B_cols + c]);
      }
      C[r * B_cols + c] = callback(sum, r, c);
    }
  }
}

#if defined(__AVX2__) && !defined(__AVX512BW__)
bool TestPrepare(int rows, int cols) {
  int size = rows * cols;
  std::mt19937 gen;
  // Go somewhat out of range too.
  std::uniform_real_distribution<float> dist(-129.0, 129.0);

  // Create array.
  float *input;
  posix_memalign((void **)&input, 64, size * sizeof(*input));
  for (int i = 0; i < size; ++i) {
    input[i] = dist(gen);
  }

  // Call Prepare
  int8_t *test;
  posix_memalign((void **)&test, 64, size * sizeof(*test));
  gemmology::PrepareB(input, test, 1, rows, cols);

  // Compute reference output.
  int8_t *quantized;
  posix_memalign((void **)&quantized, 64, size * sizeof(*quantized));
  gemmology::Quantize(input, quantized, 1, size);

  int8_t *reference;
  posix_memalign((void **)&reference, 64, size * sizeof(*reference));
  // Note this won't work for Int8/Int16 generic routines because tile sizes
  // vary.
  RearragementRef(quantized, reference, 32, 8, rows, cols);
  if (memcmp(reference, test, size * sizeof(int8_t)) != 0) {
    std::cerr << " Mismatch:\n";
    return false;
  }
  return true;
}
#endif

bool TestPrepareA(int rows, int cols) {
  std::mt19937 gen;
  // Go somewhat out of range too.
  std::uniform_real_distribution<float> dist(-2, 2);

  int size = rows * cols;
  // Create array.
  float *inputA;
  posix_memalign((void **)&inputA, 64, size * sizeof(*inputA));

  for (int i = 0; i < size; ++i) {
    inputA[i] = dist(gen);
  }

  int8_t *oldA;
  posix_memalign((void **)&oldA, 64, size * sizeof(*oldA));
  uint8_t *newA;
  posix_memalign((void **)&newA, 64, size * sizeof(*newA));
  float quant_mult = 64; // From example
  gemmology::PrepareA(inputA, oldA, quant_mult, rows, cols);
  gemmology::Shift::PrepareA(inputA, newA, quant_mult, rows, cols);
  return CompareAs(oldA, newA, rows, cols);
}

bool TestSelectColumnsB(int rows, int cols) {
  std::mt19937 gen;
  // Go somewhat out of range too.
  std::uniform_real_distribution<float> dist(-129.0, 129.0);
  int size = rows * cols;
  float *input;
  posix_memalign((void **)&input, 64, size * sizeof(*input));

  for (int i = 0; i < size; ++i) {
    input[i] = dist(gen);
  }

  int8_t *prepared;
  posix_memalign((void **)&prepared, 64, size);
  gemmology::PrepareB(input, prepared, 1, rows, cols);

  constexpr int kSelectCols = 24;
  int select_cols[kSelectCols];
  std::uniform_int_distribution<int> col_dist(0, cols - 1);
  for (auto &it : select_cols) {
    it = col_dist(gen);
  }

  int8_t *test;
  posix_memalign((void **)&test, 64, rows * kSelectCols * sizeof(*test));
  gemmology::SelectColumnsB(prepared, test, rows, select_cols,
                            select_cols + kSelectCols);

  // Select columns manually in float space.
  float *selected;
  posix_memalign((void **)&selected, 64,
                 rows * kSelectCols * sizeof(*selected));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < kSelectCols; ++c) {
      selected[c + r * kSelectCols] = input[select_cols[c] + r * cols];
    }
  }

  int8_t *ref;
  posix_memalign((void **)&ref, 64, rows * kSelectCols * sizeof(*ref));
  gemmology::PrepareB(selected, ref, 1, rows, kSelectCols);
  if (memcmp(ref, test, sizeof(int8_t) * rows * kSelectCols) != 0) {
    std::cerr << "mismatch\n";
    return false;
  }
  return true;
}

bool TestMultiplyShiftInt(int A_rows, int width, int B_cols,
                          float int_tolerance = .1, float float_tolerance = 1,
                          float MSE_float_tolerance = 0,
                          float MSE_int_tolerance = 0) {

  // Initialize A and B.
  int A_size = A_rows * width;
  int B_size = width * B_cols;
  float *A, *B, *bias;
  posix_memalign((void **)&A, 64, A_size * sizeof(*A));
  posix_memalign((void **)&B, 64, B_size * sizeof(*B));
  posix_memalign((void **)&bias, 64, B_cols * sizeof(*bias));
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < A_size; ++i) {
    A[i] = dist(gen);
  }

  for (int i = 0; i < B_size; ++i) {
    B[i] = dist(gen);
  }
  for (int i = 0; i < B_cols; ++i) {
    bias[i] = 0;
  }

  float alpha = 2.0f;
  float quant_mult = 127.0f / alpha;
  float unquant_mult = 1.0f / (quant_mult * quant_mult);

  uint8_t *A_prep;
  int8_t *B_prep;
  posix_memalign((void **)&A_prep, 64, A_size * sizeof(*A_prep));
  posix_memalign((void **)&B_prep, 64, B_size * sizeof(*B_prep));
  gemmology::Shift::PrepareA(A, A_prep, quant_mult, A_rows, width);
  gemmology::PrepareB(B, B_prep, quant_mult, width, B_cols);

  int C_size = A_rows * B_cols;
  float *test_C;
  posix_memalign((void **)&test_C, 64, C_size * sizeof(*test_C));

  /*
   * Reference float multiplication
   */
  int8_t *B_quant;
  posix_memalign((void **)&B_quant, 64, B_size * sizeof(*B_quant));
  gemmology::Quantize(B, B_quant, quant_mult, B_size);
  float *slowint_C;
  posix_memalign((void **)&slowint_C, 64, C_size * sizeof(*slowint_C));
  // Taking the original A_preparation which means A would be int8_t
  // references::Multiply(A_prep.begin(), B_quant.begin(), slowint_C.begin(),
  // A_rows, width, B_cols, [&](int32_t sum, const callbacks::OutputBufferInfo&
  // info) {
  //   return sum * unquant_mult + bias[info.col_idx];
  // });

  float *float_C;
  posix_memalign((void **)&float_C, 64, C_size * sizeof(*float_C));
  MultiplyRef(A, B, float_C, A_rows, width, B_cols,
              [&](double sum, int i, int j) {
                return static_cast<float>(sum) + bias[j];
              });

  /*
   * Multiply8 shift multiplication
   */
  // First prepare SlowInteger Bias:
  int A_prep2_size = 1 * width;
  int8_t *A_prep2;
  posix_memalign((void **)&A_prep2, 64, A_prep2_size * sizeof(*A_prep2));
  for (int i = 0; i < A_prep2_size; ++i) {
    A_prep2[i] = 1;
  }
  float *ShiftedBias;
  posix_memalign((void **)&ShiftedBias, 64, B_cols * sizeof(*ShiftedBias));
  float unquant_mult_forprep = (-1) * (alpha) * (alpha) /
                               (127.0f); // Minus one to invert add_ps later on

  MultiplyRef(A_prep2, B_quant, ShiftedBias, 1, width, B_cols,
              [&](int32_t sum, int i, int j) {
                return sum * unquant_mult_forprep + bias[j];
              });

  // Now prepare Fast integer Bias
  gemmology::Shift::PrepareBias(
      B_prep, width, B_cols,
      gemmology::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult_forprep,
                                                         bias, bias));
#if defined(_OPENMP)
  gemmology::OpenMPExecutionEngine engine;
#elif defined(GEMMOLOGY_WITH_STD_THREAD)
  gemmology::StdThreadExecutionEngine engine(4);
#else
  gemmology::SequentialExecutionEngine engine;
#endif

  gemmology::Shift::Multiply(A_prep, B_prep, A_rows, width, B_cols,
                             gemmology::callbacks::UnquantizeAndAddBiasAndWrite(
                                 unquant_mult, bias, test_C), engine);

  // Reference INT VERSION HERE with ADD127
  // Taking the original A_preparation which means A would be int8_t
  MultiplyRef(A_prep, B_quant, slowint_C, A_rows, width, B_cols,
              [&](int32_t sum, int i, int j) {
                return sum * unquant_mult + ShiftedBias[j];
              });

  return CompareMSE(float_C, slowint_C, test_C, C_size, int_tolerance,
                    float_tolerance, MSE_float_tolerance, MSE_int_tolerance);
}

bool TestPrepareBias(int rows, int cols) {
  std::mt19937 gen;
  // Go somewhat out of range too.
  std::uniform_real_distribution<float> dist(-30.0, 30.0);
  // Create array.
  int inputB_size = rows * cols;
  float *inputB;
  posix_memalign((void **)&inputB, 64, inputB_size * sizeof(*inputB));

  for (int i = 0; i < inputB_size; ++i) {
    inputB[i] = dist(gen);
  }

  float alpha = 25;
  float quant_mult = 127 / alpha;

  int8_t *B_prep;
  posix_memalign((void **)&B_prep, 64, inputB_size * sizeof(*B_prep));

  int8_t *B_quant;
  posix_memalign((void **)&B_quant, 64, inputB_size * sizeof(*B_quant));

  gemmology::PrepareB(inputB, B_prep, quant_mult, rows, cols);
  gemmology::Quantize(inputB, B_quant, quant_mult, inputB_size);

  float *inputBias;
  posix_memalign((void **)&inputBias, 64, cols * sizeof(*inputBias));

  float *goldBias;
  posix_memalign((void **)&goldBias, 64, cols * sizeof(*goldBias));

  for (int i = 0; i < cols; ++i) {
    goldBias[i] = dist(gen);
  }

  for (int i = 0; i < cols; ++i) {
    inputBias[i] = goldBias[i];
  }

  float unquant_mult_forprep = (-1) * (alpha) * (alpha) / (127.0f);

  gemmology::Shift::PrepareBias(
      B_prep, rows, cols,
      gemmology::callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult_forprep,
                                                         inputBias, inputBias));

  int A_rows = 1;
  int8_t *A_prep2;
  posix_memalign((void **)&A_prep2, 64, A_rows * rows * sizeof(*A_prep2));
  for (int i = 0; i < A_rows * rows; ++i) {
    A_prep2[i] = 1;
  }

  float *slowint_C;
  posix_memalign((void **)&slowint_C, 64, cols * sizeof(*slowint_C));

  MultiplyRef(A_prep2, B_quant, slowint_C, A_rows, rows, cols,
              [&](int32_t sum, int i, int j) {
                return sum * unquant_mult_forprep + goldBias[j];
              });
  return CompareEps(slowint_C, inputBias, cols, 0.0001f);
}
} // namespace

int main() {
#if defined(__AVX2__) && !defined(__AVX512BW__)
  if (!TestPrepare(64, 32))
    return 1;
#endif
  if (!TestSelectColumnsB(256, 256))
    return 1;
  if (!TestSelectColumnsB(512, 512))
    return 1;

  if (!TestPrepareA(64, 64))
    return 1;
  if (!TestPrepareA(256, 256))
    return 1;
  if (!TestPrepareA(512, 512))
    return 1;
  if (!TestPrepareA(2048, 256))
    return 1;

  if (!TestPrepareBias(256, 256))
    return 1;
  if (!TestPrepareBias(2048, 256))
    return 1;
  if (!TestPrepareBias(512, 512))
    return 1;

  if (!TestMultiplyShiftInt(8, 256, 256, 0.0001f, 0.54f, 0.17f, 0.0001f))
    return 1;
  if (!TestMultiplyShiftInt(8, 2048, 256, 0.0001f, 1.66f, 0.46f, 0.0001f))
    return 1;
  if (!TestMultiplyShiftInt(320, 256, 256, 0.0001f, 0.64f, 0.16f, 0.0001f))
    return 1;
  if (!TestMultiplyShiftInt(472, 256, 256, 0.0001f, 0.62f, 0.17f, 0.0001f))
    return 1;
  if (!TestMultiplyShiftInt(248, 256, 256, 0.0001f, 0.64f, 0.16f, 0.0001f))
    return 1;
  if (!TestMultiplyShiftInt(200, 256, 256, 0.0001f, 0.74f, 0.17f, 0.0001f))
    return 1;
  if (!TestMultiplyShiftInt(2, 512, 512, 0.0001f, 0.74f, 0.17f, 0.0001f))
    return 1;

  return 0;
}
