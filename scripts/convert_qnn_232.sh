#!/bin/bash
# Test QNN conversion with QAIRT SDK 2.32.6 (from cosyvoice_snpe project)
QAIRT_ROOT=/project/cosyvoice_snpe_snpe/qairt
export LD_LIBRARY_PATH=$QAIRT_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$QAIRT_ROOT/lib/python:$PYTHONPATH
PROJ=/project/cosyvoice_snpe_snpe

echo "=== SDK 2.32.6: generator_original.onnx ==="
$QAIRT_ROOT/bin/x86_64-linux-clang/qairt-converter \
  --input_network $PROJ/generator_clean.onnx \
  --output_path $PROJ/generator_232.bin \
  --backend CPU \
  --onnx_define_symbol N 1 \
  --onnx_define_symbol T 128 \
  --onnx_define_symbol T_audio 65536 \
  2>&1

echo ""
echo "=== Also try qnn-onnx-converter from 2.32.6 ==="
$QAIRT_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter \
  --input_network $PROJ/generator_clean.onnx \
  --output_path $PROJ/generator_232.cpp \
  --input_dim z "1,192,128" \
  2>&1

echo ""
echo "=== Results ==="
ls -lh $PROJ/generator_232* 2>/dev/null
echo "=== Done ==="
