#!/bin/bash
set -e
SDK26=/home/fibo/qnn_sdk_262
SDK32=/home/fibo/melotts_qnn/qairt232
FIBO=/usr/local/lib/python3.8/dist-packages/fiboaisdk
cd /home/fibo/melotts_qnn

# Extract 2.32.6 tools if not already done
if [ ! -d "$SDK32/bin" ]; then
    echo "=== Extracting SDK 2.32.6 tools ==="
    mkdir -p $SDK32
    tar -xzf qairt232_aarch64.tar.gz -C $SDK32 2>&1 | tail -1
    echo "Done"
fi

MODEL_SO=lib_arm/aarch64-ubuntu-gcc9.4/libmelotts_gen.so

echo ""
echo "=== Attempt 1: 2.32.6 context-binary-generator with DSP backend ==="
LD_LIBRARY_PATH=$SDK32/lib/aarch64-ubuntu-gcc9.4:$LD_LIBRARY_PATH \
$SDK32/bin/aarch64-ubuntu-gcc9.4/qnn-context-binary-generator \
  --model $MODEL_SO \
  --backend $SDK32/lib/aarch64-ubuntu-gcc9.4/libQnnDsp.so \
  --binary_file melotts_dsp232.bin \
  --output_dir ./ctx232 \
  2>&1 | tail -15

echo ""
echo "=== Attempt 2: 2.32.6 with HTP backend (Hexagon v68) ==="
LD_LIBRARY_PATH=$SDK32/lib/aarch64-ubuntu-gcc9.4:$LD_LIBRARY_PATH \
$SDK32/bin/aarch64-ubuntu-gcc9.4/qnn-context-binary-generator \
  --model $MODEL_SO \
  --backend $SDK32/lib/hexagon-v68/unsigned/libQnnHtpV68.so \
  --binary_file melotts_htp232.bin \
  --output_dir ./ctx232_htp \
  2>&1 | tail -15

echo ""
echo "=== Results ==="
ls -lh ctx232/melotts_dsp232.bin ctx232_htp/melotts_htp232.bin 2>/dev/null
echo "=== Done ==="
