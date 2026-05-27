#!/bin/bash
SDK=/opt/qairt/2.26.2.240911
export LD_LIBRARY_PATH=$SDK/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$SDK/lib/python:$PYTHONPATH
PROJ=/project/cosyvoice_snpe_snpe

# Generate calibration inputs if needed
mkdir -p $PROJ/calib_inputs
python3 -c "
import numpy as np
for i in range(10):
    z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
    z.tofile('$PROJ/calib_inputs/z_%04d.raw' % i)
"
echo "$PROJ/calib_inputs/z_0000.raw" > $PROJ/calib_list.txt
for i in $(seq 1 9); do
    printf -v idx "%04d" $i
    echo "$PROJ/calib_inputs/z_${idx}.raw" >> $PROJ/calib_list.txt
done

for MODEL in generator_option_b; do
    echo "=== INT8 quantize: ${MODEL}.dlc ==="
    $SDK/bin/x86_64-linux-clang/qairt-quantizer \
      --input_dlc $PROJ/${MODEL}.dlc \
      --output_dlc $PROJ/${MODEL}_int8.dlc \
      --input_list $PROJ/calib_list.txt \
      --act_bitwidth 8 \
      --weights_bitwidth 8 \
      --bias_bitwidth 32 \
      2>&1 | tail -5

    ls -lh $PROJ/${MODEL}_int8.dlc 2>/dev/null
done

echo ""
echo "=== Also quantize generator_original INT8 ==="
$SDK/bin/x86_64-linux-clang/qairt-quantizer \
  --input_dlc $PROJ/generator_test.dlc \
  --output_dlc $PROJ/generator_original_int8.dlc \
  --input_list $PROJ/calib_list.txt \
  --act_bitwidth 8 \
  --weights_bitwidth 8 \
  --bias_bitwidth 32 \
  2>&1 | tail -5

ls -lh $PROJ/generator_original_int8.dlc $PROJ/generator_option_b_int8.dlc 2>/dev/null
echo "Done"
