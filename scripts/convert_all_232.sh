#!/bin/bash
QAIRT_ROOT=/project/cosyvoice_snpe_snpe/qairt
export LD_LIBRARY_PATH=$QAIRT_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$QAIRT_ROOT/lib/python:$PYTHONPATH
PROJ=/project/cosyvoice_snpe_snpe

echo "=== 1. Convert generator_original.onnx ==="
$QAIRT_ROOT/bin/x86_64-linux-clang/qairt-converter \
  --input_network $PROJ/generator_clean.onnx \
  --output_path $PROJ/generator_original.bin \
  --target_backend CPU \
  --onnx_define_symbol N 1 \
  --onnx_define_symbol T 128 \
  --onnx_define_symbol T_audio 65536 \
  2>&1 | grep -E 'INFO_CONVERSION|ERROR|WARNING_WRITE'

echo "=== 2. Convert generator_option_b.onnx ==="
$QAIRT_ROOT/bin/x86_64-linux-clang/qairt-converter \
  --input_network $PROJ/generator_option_b.onnx \
  --output_path $PROJ/generator_option_b.bin \
  --target_backend CPU \
  --onnx_define_symbol N 1 \
  --onnx_define_symbol T 128 \
  --onnx_define_symbol T_audio 65536 \
  2>&1 | grep -E 'INFO_CONVERSION|ERROR|WARNING_WRITE'

echo ""
echo "=== Results ==="
ls -lh $PROJ/generator_*.bin 2>/dev/null
echo "=== Done ==="
