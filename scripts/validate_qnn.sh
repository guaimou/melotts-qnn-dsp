#!/bin/bash
QAIRT_ROOT=/project/cosyvoice_snpe_snpe/qairt
export LD_LIBRARY_PATH=$QAIRT_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
PROJ=/project/cosyvoice_snpe_snpe
BACKEND=$QAIRT_ROOT/lib/x86_64-linux-clang/libQnnCpu.so

echo "=== Prepare test input ==="
python3 -c "
import numpy as np
z = np.random.randn(1, 192, 128).astype(np.float32)
z.tofile('$PROJ/z_input.raw')
print(f'z shape: {z.shape}, mean={z.mean():.4f}, std={z.std():.4f}')
"

echo "z:=z_input.raw" > $PROJ/input_list.txt

echo ""
echo "=== Run QNN inference (original) ==="
$QAIRT_ROOT/bin/x86_64-linux-clang/qnn-net-run \
  --model $PROJ/generator_original.bin \
  --backend $BACKEND \
  --input_list $PROJ/input_list.txt \
  --output_dir $PROJ/output_original \
  2>&1 | tail -10

echo ""
echo "=== Run QNN inference (option_b) ==="
$QAIRT_ROOT/bin/x86_64-linux-clang/qnn-net-run \
  --model $PROJ/generator_option_b.bin \
  --backend $BACKEND \
  --input_list $PROJ/input_list.txt \
  --output_dir $PROJ/output_option_b \
  2>&1 | tail -10

echo ""
echo "=== Compare outputs ==="
python3 -c "
import numpy as np
orig = np.fromfile('$PROJ/output_original/Result_0/y.raw', dtype=np.float32)
optb = np.fromfile('$PROJ/output_option_b/Result_0/y.raw', dtype=np.float32)
print(f'Original: shape={orig.shape}, mean={orig.mean():.4f}, std={orig.std():.4f}, min={orig.min():.4f}, max={orig.max():.4f}')
print(f'Option B: shape={optb.shape}, mean={optb.mean():.4f}, std={optb.std():.4f}, min={optb.min():.4f}, max={optb.max():.4f}')
"

echo ""
echo "=== Output files ==="
ls -lh $PROJ/output_original/Result_0/ $PROJ/output_option_b/Result_0/ 2>/dev/null
echo "=== Done ==="
