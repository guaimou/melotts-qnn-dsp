#!/bin/bash
SDK=/opt/qairt/2.26.2.240911
export LD_LIBRARY_PATH=$SDK/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$SDK/lib/python:$PYTHONPATH
PROJ=/project/cosyvoice_snpe_snpe

echo "=== 1. qairt-converter (CPU backend, no soc_model) ==="
$SDK/bin/x86_64-linux-clang/qairt-converter \
  --input_network $PROJ/generator_option_b.onnx \
  --output_path $PROJ/gen262_optb.cpp \
  --backend CPU \
  --onnx_define_symbol N 1 \
  --onnx_define_symbol T 128 \
  --onnx_define_symbol T_audio 65536 \
  2>&1 | grep -E "INFO_CONVERSION|INFO_WRITE|ERROR"

echo ""
echo "=== Check format ==="
xxd $PROJ/gen262_optb.cpp 2>/dev/null | head -3
ls -lh $PROJ/gen262_optb.cpp 2>/dev/null

echo ""
echo "=== 2. qnn-onnx-converter (original 2.26.2) ==="
$SDK/bin/x86_64-linux-clang/qnn-onnx-converter \
  --input_network $PROJ/generator_option_b.onnx \
  --output_path $PROJ/gen262_qnn.cpp \
  --input_dim z "1,192,128" \
  2>&1 | grep -E "INFO_CONVERSION|ERROR|success|written"

echo ""
xxd $PROJ/gen262_qnn.cpp 2>/dev/null | head -3
ls -lh $PROJ/gen262_qnn.cpp 2>/dev/null

echo ""
echo "=== 3. Check supported soc models ==="
python3 -c "
from qti.aisw.converters.common.backend_awareness import BackendInfo
for soc in ['8550', '6490', '8Gen2', '8Gen3', 'SM8550', 'SM8650']:
    try:
        bi = BackendInfo.get_instance('DSP', soc)
        print(f'  {soc}: supported')
    except Exception as e:
        print(f'  {soc}: NOT supported')
" 2>&1
