#!/bin/bash
SDK=/opt/qairt/2.26.2.240911
export LD_LIBRARY_PATH=$SDK/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
PROJ=/project/cosyvoice_snpe_snpe

echo "=== 1. Generate model.so ==="
$SDK/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c $PROJ/gen262_clean.cpp \
  -b $PROJ/gen262_clean.bin \
  -t x86_64-linux-clang \
  -l melotts_gen \
  -o $PROJ/lib262 \
  2>&1 | tail -10

echo ""
echo "=== Check .so ==="
find $PROJ/lib262 -name '*.so' -ls 2>/dev/null
echo ""

echo "=== 2. Check supported target archs ==="
ls -d $SDK/lib/hexagon-* 2>/dev/null

echo ""
echo "=== Hexagon v68 DSP backend ==="
ls $SDK/lib/hexagon-v68/unsigned/libQnnDsp*.so 2>/dev/null
ls $SDK/lib/aarch64-android/libQnnDsp.so 2>/dev/null

echo ""
echo "=== 3. Try context binary (CPU first, then DSP) ==="
# CPU test
if [ -f $PROJ/lib262/x86_64-linux-clang/libmelotts_gen.so ]; then
    $SDK/bin/x86_64-linux-clang/qnn-context-binary-generator \
      --model $PROJ/lib262/x86_64-linux-clang/libmelotts_gen.so \
      --backend $SDK/lib/x86_64-linux-clang/libQnnCpu.so \
      --binary_file melotts_gen_cpu.bin \
      --output_dir $PROJ/ctx262 \
      2>&1 | tail -10

    echo ""
    ls -lh $PROJ/ctx262/melotts_gen_cpu.bin 2>/dev/null
fi
