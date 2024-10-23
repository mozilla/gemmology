##                  ##
# Configuration part #

XSIMD_INCLUDE_DIR=/usr
SDE64=sde64

# ARM is disabled by default
ARM_CXX=:
ARM_QEMU=:
ARM64_CXX=:
ARM64_QEMU=:
######################


GEMMOLOGY_CPPFLAGS= -I.. -I$(XSIMD_INCLUDE_DIR) $(CPPFLAGS)
GEMMOLOGY_CXXFLAGS= -std=c++17 $(CXXFLAGS)

##

all:check

check:check.avx2 check.avxvnni check.sse4 check.ssse3 check.sse2 check.avx512 check.avx512vnni check.neon check.neon64 check.neon64+i8mm check.thread $(if $(NOOMP),,check.omp)

clean:clean.avx2 clean.avxvnni clean.sse4 clean.ssse3 clean.sse2 clean.avx512 clean.avx512vnni clean.neon clean.neon64 clean.neon64+i8mm clean.thread $(if $(NOOMP),,clean.omp)

GEMMOLOGY_NOASAN_CXXFLAGS=$(filter-out -fsanitize=address,$(GEMMOLOGY_CXXFLAGS))

# AVX512VNNI
test_transpose.avx512vnni: test_transpose.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mavx512vnni -mavx512bw -mavx512f -mavx512dq -mavx512cd

test_prepare_b_transposed.avx512vnni: test_prepare_b_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mavx512vnni -mavx512bw -mavx512f -mavx512dq -mavx512cd

test_prepare_b_quantized_transposed.avx512vnni: test_prepare_b_quantized_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@  -mavx512vnni -mavx512bw -mavx512f -mavx512dq -mavx512cd

test_multiply.avx512vnni:test_multiply.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@  -mavx512vnni -mavx512bw -mavx512f -mavx512dq -mavx512cd

test_quantize.avx512vnni:test_quantize.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@  -mavx512vnni -mavx512bw -mavx512f -mavx512dq -mavx512cd

check.avx512vnni:test_prepare_b_transposed.avx512vnni test_prepare_b_quantized_transposed.avx512vnni test_multiply.avx512vnni test_quantize.avx512vnni test_transpose.avx512vnni
	$(SDE64) -icx -- ./test_transpose.avx512vnni
	$(SDE64) -icx -- ./test_prepare_b_transposed.avx512vnni
	$(SDE64) -icx -- ./test_prepare_b_quantized_transposed.avx512vnni
	$(SDE64) -icx -- ./test_quantize.avx512vnni
	$(SDE64) -icx -- ./test_multiply.avx512vnni

clean.avx512vnni:
	$(RM) test_prepare_b_transposed.avx512vnni test_prepare_b_quantized_transposed.avx512vnni test_multiply.avx512vnni test_quantize.avx512vnni test_transpose.avx512


# AVX512
test_transpose.avx512: test_transpose.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mavx512bw -mavx512f -mavx512dq -mavx512cd

test_prepare_b_transposed.avx512: test_prepare_b_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mavx512bw -mavx512f -mavx512dq -mavx512cd

test_prepare_b_quantized_transposed.avx512: test_prepare_b_quantized_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mavx512bw -mavx512f -mavx512dq -mavx512cd

test_multiply.avx512:test_multiply.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mavx512bw -mavx512f -mavx512dq -mavx512cd

test_quantize.avx512:test_quantize.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mavx512bw -mavx512f -mavx512dq -mavx512cd

check.avx512:test_prepare_b_transposed.avx512 test_prepare_b_quantized_transposed.avx512 test_multiply.avx512 test_quantize.avx512 test_transpose.avx512
	$(SDE64) -skx -- ./test_transpose.avx512
	$(SDE64) -skx -- ./test_prepare_b_transposed.avx512
	$(SDE64) -skx -- ./test_prepare_b_quantized_transposed.avx512
	$(SDE64) -skx -- ./test_quantize.avx512
	$(SDE64) -skx -- ./test_multiply.avx512

clean.avx512:
	$(RM) test_prepare_b_transposed.avx512 test_prepare_b_quantized_transposed.avx512 test_multiply.avx512 test_quantize.avx512 test_transpose.avx512

# AVXVNNI
test_transpose.avxvnni: test_transpose.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mavxvnni

test_prepare_b_transposed.avxvnni: test_prepare_b_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mavxvnni

test_prepare_b_quantized_transposed.avxvnni: test_prepare_b_quantized_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mavxvnni

test_multiply.avxvnni:test_multiply.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mavxvnni

test_quantize.avxvnni:test_quantize.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mavxvnni

check.avxvnni:test_prepare_b_transposed.avxvnni test_prepare_b_quantized_transposed.avxvnni test_multiply.avxvnni test_quantize.avxvnni test_transpose.avxvnni
	$(SDE64) -adl -- ./test_transpose.avxvnni
	$(SDE64) -adl -- ./test_prepare_b_transposed.avxvnni
	$(SDE64) -adl -- ./test_prepare_b_quantized_transposed.avxvnni
	$(SDE64) -adl -- ./test_quantize.avxvnni
	$(SDE64) -adl -- ./test_multiply.avxvnni

clean.avxvnni:
	$(RM) test_prepare_b_transposed.avxvnni test_prepare_b_quantized_transposed.avxvnni test_multiply.avxvnni test_quantize.avxvnni test_transpose.avxvnni


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

# Neon
test_prepare_b_transposed.neon: test_prepare_b_transposed.cpp ../gemmology.h Makefile
	$(ARM_CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mfpu=neon

test_prepare_b_quantized_transposed.neon: test_prepare_b_quantized_transposed.cpp ../gemmology.h Makefile
	$(ARM_CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mfpu=neon

test_multiply.neon:test_multiply.cpp Makefile ../gemmology.h
	$(ARM_CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mfpu=neon

test_quantize.neon:test_quantize.cpp Makefile ../gemmology.h
	$(ARM_CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -mfpu=neon

check.neon:test_prepare_b_transposed.neon test_prepare_b_quantized_transposed.neon test_multiply.neon test_quantize.neon
	$(ARM_QEMU) ./test_prepare_b_transposed.neon
	$(ARM_QEMU) ./test_prepare_b_quantized_transposed.neon
	$(ARM_QEMU) ./test_quantize.neon
	$(ARM_QEMU) ./test_multiply.neon

clean.neon:
	$(RM) test_prepare_b_transposed.neon test_prepare_b_quantized_transposed.neon test_multiply.neon test_quantize.neon test_transpose.neon

# Neon64
test_prepare_b_transposed.neon64: test_prepare_b_transposed.cpp ../gemmology.h Makefile
	$(ARM64_CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@

test_prepare_b_quantized_transposed.neon64: test_prepare_b_quantized_transposed.cpp ../gemmology.h Makefile
	$(ARM64_CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@

test_multiply.neon64:test_multiply.cpp Makefile ../gemmology.h
	$(ARM64_CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@

test_quantize.neon64:test_quantize.cpp Makefile ../gemmology.h
	$(ARM64_CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@

check.neon64:test_prepare_b_transposed.neon64 test_prepare_b_quantized_transposed.neon64 test_multiply.neon64 test_quantize.neon64
	$(ARM64_QEMU) ./test_prepare_b_transposed.neon64
	$(ARM64_QEMU) ./test_prepare_b_quantized_transposed.neon64
	$(ARM64_QEMU) ./test_quantize.neon64
	$(ARM64_QEMU) ./test_multiply.neon64

clean.neon64:
	$(RM) test_prepare_b_transposed.neon64 test_prepare_b_quantized_transposed.neon64 test_multiply.neon64 test_quantize.neon64 test_transpose.neon64

# Neon64+i8mm
test_prepare_b_transposed.neon64+i8mm: test_prepare_b_transposed.cpp ../gemmology.h Makefile
	$(ARM64_CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -march=armv8.4-a+i8mm

test_prepare_b_quantized_transposed.neon64+i8mm: test_prepare_b_quantized_transposed.cpp ../gemmology.h Makefile
	$(ARM64_CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -march=armv8.4-a+i8mm

test_multiply.neon64+i8mm:test_multiply.cpp Makefile ../gemmology.h
	$(ARM64_CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -march=armv8.4-a+i8mm

test_quantize.neon64+i8mm:test_quantize.cpp Makefile ../gemmology.h
	$(ARM64_CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_NOASAN_CXXFLAGS) $< -o $@ -march=armv8.4-a+i8mm

check.neon64+i8mm:test_prepare_b_transposed.neon64+i8mm test_prepare_b_quantized_transposed.neon64+i8mm test_multiply.neon64+i8mm test_quantize.neon64+i8mm
	$(ARM64_QEMU) ./test_prepare_b_transposed.neon64+i8mm
	$(ARM64_QEMU) ./test_prepare_b_quantized_transposed.neon64+i8mm
	$(ARM64_QEMU) ./test_quantize.neon64+i8mm
	$(ARM64_QEMU) ./test_multiply.neon64+i8mm

clean.neon64+i8mm:
	$(RM) test_prepare_b_transposed.neon64+i8mm test_prepare_b_quantized_transposed.neon64+i8mm test_multiply.neon64+i8mm test_quantize.neon64+i8mm test_transpose.neon64+i8mm

# OpenMP
test_transpose.omp: test_transpose.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -fopenmp

test_prepare_b_transposed.omp: test_prepare_b_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -fopenmp

test_prepare_b_quantized_transposed.omp: test_prepare_b_quantized_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -fopenmp

test_multiply.omp:test_multiply.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -fopenmp

test_quantize.omp:test_quantize.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -fopenmp

check.omp:test_prepare_b_transposed.omp test_prepare_b_quantized_transposed.omp test_multiply.omp test_quantize.omp test_transpose.omp
	./test_transpose.omp
	./test_prepare_b_transposed.omp
	./test_prepare_b_quantized_transposed.omp
	./test_quantize.omp
	./test_multiply.omp

clean.omp:
	$(RM) test_prepare_b_transposed.omp test_prepare_b_quantized_transposed.omp test_multiply.omp test_quantize.omp test_transpose.omp


# Thread
test_transpose.thread: test_transpose.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -DGEMMOLOGY_WITH_STD_THREAD

test_prepare_b_transposed.thread: test_prepare_b_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -DGEMMOLOGY_WITH_STD_THREAD

test_prepare_b_quantized_transposed.thread: test_prepare_b_quantized_transposed.cpp ../gemmology.h Makefile
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -DGEMMOLOGY_WITH_STD_THREAD

test_multiply.thread:test_multiply.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -DGEMMOLOGY_WITH_STD_THREAD

test_quantize.thread:test_quantize.cpp Makefile ../gemmology.h
	$(CXX) $(GEMMOLOGY_CPPFLAGS) $(GEMMOLOGY_CXXFLAGS) $< -o $@ -DGEMMOLOGY_WITH_STD_THREAD

check.thread:test_prepare_b_transposed.thread test_prepare_b_quantized_transposed.thread test_multiply.thread test_quantize.thread test_transpose.thread
	./test_transpose.thread
	./test_prepare_b_transposed.thread
	./test_prepare_b_quantized_transposed.thread
	./test_quantize.thread
	./test_multiply.thread

clean.thread:
	$(RM) test_prepare_b_transposed.thread test_prepare_b_quantized_transposed.thread test_multiply.thread test_quantize.thread test_transpose.thread
