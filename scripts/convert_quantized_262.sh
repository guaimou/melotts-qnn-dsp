#!/bin/bash
SDK=/opt/qairt/2.26.2.240911
export LD_LIBRARY_PATH=$SDK/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$SDK/lib/python:$PYTHONPATH
PROJ=/project/cosyvoice_snpe_snpe
PY=/opt/snpe_python310_env/bin/python

# Generate calibration inputs
mkdir -p $PROJ/calib_inputs
python3 -c "
import numpy as np
for i in range(20):
    z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5
    z.tofile('$PROJ/calib_inputs/z_%04d.raw' % i)
" 2>/dev/null

# Create input_list with calibration paths
echo "$PROJ/calib_inputs/z_0000.raw" > $PROJ/calib_list.txt
for i in $(seq 1 19); do
    printf -v idx "%04d" $i
    echo "$PROJ/calib_inputs/z_${idx}.raw" >> $PROJ/calib_list.txt
done

echo "=== qnn-onnx-converter with INT16 quantization ==="
$PY $SDK/bin/x86_64-linux-clang/qnn-onnx-converter \
  --input_network $PROJ/generator_option_b.onnx \
  --output_path $PROJ/gen262_int16.cpp \
  --input_dim z "1,192,128" \
  --input_list $PROJ/calib_list.txt \
  --act_bitwidth 16 \
  --weights_bitwidth 16 \
  --bias_bitwidth 32 \
  --no_simplification \
  2>&1 | tail -10

echo ""
echo "=== Check output ==="
ls -lh $PROJ/gen262_int16.cpp $PROJ/gen262_int16.bin 2>/dev/null
xxd $PROJ/gen262_int16.cpp 2>/dev/null | head -2

echo ""
echo "=== Also try INT8 quantization ==="
$PY $SDK/bin/x86_64-linux-clang/qnn-onnx-converter \
  --input_network $PROJ/generator_option_b.onnx \
  --output_path $PROJ/gen262_int8.cpp \
  --input_dim z "1,192,128" \
  --input_list $PROJ/calib_list.txt \
  --act_bitwidth 8 \
  --weights_bitwidth 8 \
  --bias_bitwidth 32 \
  --no_simplification \
  2>&1 | tail -10

echo ""
ls -lh $PROJ/gen262_int8.cpp $PROJ/gen262_int8.bin 2>/dev/null
echo "Done"
