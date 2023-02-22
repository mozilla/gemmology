## Configuration part
XSIMD_INCLUDE_DIR=/usr
SDE64=sde64
##


GEMMOLOGY_CPPFLAGS= -I.. -I$(XSIMD_INCLUDE_DIR) $(CPPFLAGS)
GEMMOLOGY_CXXFLAGS= -std=c++17 $(CXXFLAGS)

##

all:check

check:check.avx2 check.sse4 check.ssse3 check.sse2 check.avx512

clean:clean.avx2 clean.sse4 clean.ssse3 clean.sse2 clean.avx512

# AVX512
test_transpose.avx512: test_transpose.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -mavx512bw -mavx512f -mavx512dq -mavx512cd

test_prepare_b_transposed.avx512: test_prepare_b_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -mavx512bw -mavx512f -mavx512dq -mavx512cd

test_prepare_b_quantized_transposed.avx512: test_prepare_b_quantized_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -mavx512bw -mavx512f -mavx512dq -mavx512cd

test_multiply.avx512:test_multiply.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -mavx512bw -mavx512f -mavx512dq -mavx512cd

test_quantize.avx512:test_quantize.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -mavx512bw -mavx512f -mavx512dq -mavx512cd

check.avx512:test_prepare_b_transposed.avx512 test_prepare_b_quantized_transposed.avx512 test_multiply.avx512 test_quantize.avx512 test_transpose.avx512
	$(SDE64) -skx -- ./test_transpose.avx512
	$(SDE64) -skx -- ./test_prepare_b_transposed.avx512
	$(SDE64) -skx -- ./test_prepare_b_quantized_transposed.avx512
	$(SDE64) -skx -- ./test_quantize.avx512
	$(SDE64) -skx -- ./test_multiply.avx512

clean.avx512:
	$(RM) test_prepare_b_transposed.avx512 test_prepare_b_quantized_transposed.avx512 test_multiply.avx512 test_quantize.avx512 test_transpose.avx512

# AVX2
test_transpose.avx2: test_transpose.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -mavx2

test_prepare_b_transposed.avx2: test_prepare_b_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -mavx2

test_prepare_b_quantized_transposed.avx2: test_prepare_b_quantized_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -mavx2

test_multiply.avx2:test_multiply.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -mavx2

test_quantize.avx2:test_quantize.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -mavx2

check.avx2:test_prepare_b_transposed.avx2 test_prepare_b_quantized_transposed.avx2 test_multiply.avx2 test_quantize.avx2 test_transpose.avx2
	./test_transpose.avx2
	./test_prepare_b_transposed.avx2
	./test_prepare_b_quantized_transposed.avx2
	./test_quantize.avx2
	./test_multiply.avx2

clean.avx2:
	$(RM) test_prepare_b_transposed.avx2 test_prepare_b_quantized_transposed.avx2 test_multiply.avx2 test_quantize.avx2 test_transpose.avx2

# SSE4.2
test_transpose.sse4: test_transpose.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -msse4.2

test_prepare_b_transposed.sse4: test_prepare_b_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -msse4.2

test_prepare_b_quantized_transposed.sse4: test_prepare_b_quantized_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -msse4.2

test_multiply.sse4:test_multiply.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -msse4.2

test_quantize.sse4:test_quantize.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -msse4.2

check.sse4:test_prepare_b_transposed.sse4 test_prepare_b_quantized_transposed.sse4 test_multiply.sse4 test_quantize.sse4 test_transpose.sse4
	./test_transpose.sse4
	./test_prepare_b_transposed.sse4
	./test_prepare_b_quantized_transposed.sse4
	./test_quantize.sse4
	./test_multiply.sse4

clean.sse4:
	$(RM) test_prepare_b_transposed.sse4 test_prepare_b_quantized_transposed.sse4 test_multiply.sse4 test_quantize.sse4 test_transpose.sse4

# SSSE3
test_transpose.ssse3: test_transpose.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -mssse3

test_prepare_b_transposed.ssse3: test_prepare_b_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -mssse3

test_prepare_b_quantized_transposed.ssse3: test_prepare_b_quantized_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -mssse3

test_multiply.ssse3:test_multiply.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -mssse3

test_quantize.ssse3:test_quantize.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -mssse3

check.ssse3:test_prepare_b_transposed.ssse3 test_prepare_b_quantized_transposed.ssse3 test_multiply.ssse3 test_quantize.ssse3 test_transpose.ssse3
	./test_transpose.ssse3
	./test_prepare_b_transposed.ssse3
	./test_prepare_b_quantized_transposed.ssse3
	./test_quantize.ssse3
	./test_multiply.ssse3

clean.ssse3:
	$(RM) test_prepare_b_transposed.ssse3 test_prepare_b_quantized_transposed.ssse3 test_multiply.ssse3 test_quantize.ssse3 test_transpose.ssse3

# SSE2
test_transpose.sse2: test_transpose.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -msse2

test_prepare_b_transposed.sse2: test_prepare_b_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -msse2

test_prepare_b_quantized_transposed.sse2: test_prepare_b_quantized_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -msse2

test_multiply.sse2:test_multiply.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -msse2

test_quantize.sse2:test_quantize.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -msse2

check.sse2:test_prepare_b_transposed.sse2 test_prepare_b_quantized_transposed.sse2 test_multiply.sse2 test_quantize.sse2 test_transpose.sse2
	./test_transpose.sse2
	./test_prepare_b_transposed.sse2
	./test_prepare_b_quantized_transposed.sse2
	./test_quantize.sse2
	./test_multiply.sse2

clean.sse2:
	$(RM) test_prepare_b_transposed.sse2 test_prepare_b_quantized_transposed.sse2 test_multiply.sse2 test_quantize.sse2 test_transpose.sse2
