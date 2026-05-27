#!/bin/bash
QAIRT_ROOT=/project/cosyvoice_snpe_snpe/qairt
export LD_LIBRARY_PATH=$QAIRT_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
PROJ=/project/cosyvoice_snpe_snpe

echo "=== Check qairt-converter output format options ==="
$QAIRT_ROOT/bin/x86_64-linux-clang/qairt-converter --help 2>&1 | grep -iE 'output|format|cpp|bin|context' | head -20

echo ""
echo "=== Try direct qairt-converter to .so ==="
$QAIRT_ROOT/bin/x86_64-linux-clang/qairt-converter \
  --input_network $PROJ/generator_clean.onnx \
  --output_path $PROJ/libgenerator_original.so \
  --target_backend CPU \
  --onnx_define_symbol N 1 \
  --onnx_define_symbol T 128 \
  --onnx_define_symbol T_audio 65536 \
  2>&1 | grep -E 'INFO_CONVERSION|INFO_WRITE|ERROR'

echo ""
echo "=== Results ==="
ls -lh $PROJ/libgenerator_*.so 2>/dev/null
echo "=== Done ==="
