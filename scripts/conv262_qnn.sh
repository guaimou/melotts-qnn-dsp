#!/bin/bash
SDK=/opt/qairt/2.26.2.240911
export LD_LIBRARY_PATH=$SDK/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$SDK/lib/python:$PYTHONPATH
PROJ=/project/cosyvoice_snpe_snpe
PY=/opt/snpe_python310_env/bin/python

$PY -m pip install pandas -q 2>&1 | tail -1

echo "=== qnn-onnx-converter 2.26.2 ==="
$PY $SDK/bin/x86_64-linux-clang/qnn-onnx-converter \
  --input_network $PROJ/generator_clean.onnx \
  --output_path $PROJ/gen262_clean.cpp \
  --input_dim z "1,192,128" \
  2>&1 | tail -20

echo ""
echo "=== Output ==="
ls -lh $PROJ/gen262_clean.cpp 2>/dev/null
xxd $PROJ/gen262_clean.cpp 2>/dev/null | head -3
