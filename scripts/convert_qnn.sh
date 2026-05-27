#!/bin/bash
source /opt/2.29.0.241129/bin/envsetup.sh
export PATH=/opt/snpe_python310_env/bin:$PATH
PROJ=/project/cosyvoice_snpe_snpe

echo "=== QAIRT: generator_clean.onnx (no g) ==="
qairt-converter \
  --input_network $PROJ/generator_clean.onnx \
  --output_path $PROJ/generator_qnn.bin \
  --backend CPU \
  --onnx_define_symbol N 1 \
  --onnx_define_symbol T 128 \
  --onnx_define_symbol T_audio 65536 \
  2>&1

echo ""
echo "=== QAIRT: dp_clean.onnx (no g) ==="
qairt-converter \
  --input_network $PROJ/dp_clean.onnx \
  --output_path $PROJ/dp_qnn.bin \
  --backend CPU \
  --onnx_define_symbol N 1 \
  --onnx_define_symbol L 64 \
  2>&1

echo ""
echo "=== Results ==="
ls -lh $PROJ/generator_qnn* $PROJ/dp_qnn* 2>/dev/null
echo "=== Done ==="
