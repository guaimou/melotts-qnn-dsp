#!/bin/bash
QAIRT_ROOT=/project/cosyvoice_snpe_snpe/qairt
export LD_LIBRARY_PATH=$QAIRT_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$QAIRT_ROOT/lib/python:$PYTHONPATH
PROJ=/project/cosyvoice_snpe_snpe

echo "=== Generate calibration inputs ==="
CALIB_DIR=$PROJ/calib_inputs
mkdir -p $CALIB_DIR
python3 -c "
import numpy as np
# Generate 10 representative z-latent inputs for calibration
for i in range(10):
    z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
    z.tofile(f'$CALIB_DIR/z_{i:04d}.raw')
print('Generated 10 calibration inputs, shape=(1,192,128)')
"

echo "$CALIB_DIR/z_0000.raw" > $PROJ/calib_list.txt
for i in $(seq 1 9); do
    idx=$(printf "%04d" $i)
    echo "$CALIB_DIR/z_${idx}.raw" >> $PROJ/calib_list.txt
done

echo "=== Quantize generator_original.dlc -> INT16 ==="
$QAIRT_ROOT/bin/x86_64-linux-clang/qairt-quantizer \
  --input_dlc $PROJ/generator_test.dlc \
  --output_dlc $PROJ/generator_int16.dlc \
  --input_list $PROJ/calib_list.txt \
  --act_bitwidth 16 \
  --weights_bitwidth 16 \
  --bias_bitwidth 32 \
  --target_backend DSP \
  --target_soc_model 8550 \
  2>&1 | grep -E 'INFO|ERROR|success|quantiz|written|bitwidth'

echo ""
echo "=== Also quantize Option B ==="
$QAIRT_ROOT/bin/x86_64-linux-clang/qairt-quantizer \
  --input_dlc $PROJ/generator_option_b.dlc \
  --output_dlc $PROJ/generator_option_b_int16.dlc \
  --input_list $PROJ/calib_list.txt \
  --act_bitwidth 16 \
  --weights_bitwidth 16 \
  --bias_bitwidth 32 \
  --target_backend DSP \
  --target_soc_model 8550 \
  2>&1 | grep -E 'INFO|ERROR|success|quantiz|written|bitwidth'

echo ""
echo "=== Results ==="
ls -lh $PROJ/generator_*int16*.dlc 2>/dev/null
echo "=== Done ==="
