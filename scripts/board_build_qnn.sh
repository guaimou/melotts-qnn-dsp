#!/bin/bash
set -e
SDK=/home/fibo/qnn_sdk_262
FIBO=/usr/local/lib/python3.8/dist-packages/fiboaisdk
cd /home/fibo/melotts_qnn

echo "=== Generate DSP context binary ==="
LD_LIBRARY_PATH=$SDK/lib/aarch64-ubuntu-gcc9.4:$LD_LIBRARY_PATH \
$SDK/bin/aarch64-ubuntu-gcc9.4/qnn-context-binary-generator \
  --model lib_arm/aarch64-ubuntu-gcc9.4/libmelotts_gen.so \
  --backend $FIBO/libQnnDsp.so \
  --binary_file melotts_gen_dsp.bin \
  --output_dir ./ctx262 \
  2>&1 | tail -15

echo ""
echo "=== Results ==="
ls -lh ctx262/melotts_gen_dsp.bin 2>/dev/null
echo ""

echo "=== Test with QNN net-run (CPU first) ==="
python3 -c "
import numpy as np
z = np.random.randn(1, 192, 128).astype(np.float32)
z.tofile('/home/fibo/melotts_qnn/z_input.raw')
"
echo "z:=$(pwd)/z_input.raw" > input_qnn.txt

LD_LIBRARY_PATH=$SDK/lib/aarch64-ubuntu-gcc9.4:$LD_LIBRARY_PATH \
$SDK/bin/aarch64-ubuntu-gcc9.4/qnn-net-run \
  --model lib_arm/aarch64-ubuntu-gcc9.4/libmelotts_gen.so \
  --backend $SDK/lib/aarch64-ubuntu-gcc9.4/libQnnCpu.so \
  --input_list input_qnn.txt \
  --output_dir ./output_qnn \
  2>&1 | tail -10

echo ""
python3 -c "
import numpy as np, os
for f in os.listdir('/home/fibo/melotts_qnn/output_qnn/Result_0'):
    if f.endswith('.raw'):
        data = np.fromfile('/home/fibo/melotts_qnn/output_qnn/Result_0/' + f, dtype=np.float32)
        print('Output: shape %s, mean=%.4f, std=%.4f' % (str(data.shape), data.mean(), data.std()))
"
