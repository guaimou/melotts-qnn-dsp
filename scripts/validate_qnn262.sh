#!/bin/bash
SDK=/opt/qairt/2.26.2.240911
export LD_LIBRARY_PATH=$SDK/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
PROJ=/project/cosyvoice_snpe_snpe

echo "=== 1. Test QNN CPU inference ==="
python3 -c "
import numpy as np
z = np.random.randn(1, 192, 128).astype(np.float32)
z.tofile('$PROJ/z_input.raw')
print(f'z: shape (1,192,128), mean={z.mean():.4f}')
"
echo "z:=z_input.raw" > $PROJ/input262.txt

$SDK/bin/x86_64-linux-clang/qnn-net-run \
  --model $PROJ/lib262/x86_64-linux-clang/libmelotts_gen.so \
  --backend $SDK/lib/x86_64-linux-clang/libQnnCpu.so \
  --input_list $PROJ/input262.txt \
  --output_dir $PROJ/output262 \
  2>&1 | tail -10

echo ""
echo "=== Check output ==="
python3 -c "
import numpy as np, os
out_path = '$PROJ/output262/Result_0/y.raw'
if os.path.exists(out_path):
    y = np.fromfile(out_path, dtype=np.float32)
    print(f'y: shape {y.shape}, mean={y.mean():.4f}, std={y.std():.4f}')
    print(f'  Expected 65536 samples for T=128, hop=512')
else:
    print('No output found')
    os.system('ls -R $PROJ/output262/ 2>/dev/null')
"

echo ""
echo "=== 2. Try aarch64 cross-compile ==="
$SDK/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c $PROJ/gen262_clean.cpp \
  -b $PROJ/gen262_clean.bin \
  -t aarch64-android \
  -l melotts_gen \
  -o $PROJ/lib262_arm \
  2>&1 | tail -5

echo ""
find $PROJ/lib262_arm -name '*.so' -ls 2>/dev/null
