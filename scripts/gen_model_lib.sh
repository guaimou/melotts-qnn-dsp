#!/bin/bash
QAIRT_ROOT=/project/cosyvoice_snpe_snpe/qairt
export LD_LIBRARY_PATH=$QAIRT_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$QAIRT_ROOT/lib/python:$PYTHONPATH
PROJ=/project/cosyvoice_snpe_snpe

for MODEL in generator_original generator_option_b; do
    echo "=== qairt-converter: $MODEL -> .cpp ==="
    $QAIRT_ROOT/bin/x86_64-linux-clang/qairt-converter \
      --input_network $PROJ/${MODEL}.onnx \
      --output_path $PROJ/${MODEL}.cpp \
      --target_backend CPU \
      --onnx_define_symbol N 1 \
      --onnx_define_symbol T 128 \
      --onnx_define_symbol T_audio 65536 \
      2>&1 | grep -E 'INFO_CONVERSION|INFO_WRITE|ERROR'

    echo "=== qnn-model-lib-generator: $MODEL -> .so ==="
    $QAIRT_ROOT/bin/x86_64-linux-clang/qnn-model-lib-generator \
      -c $PROJ/${MODEL}.cpp \
      -b $PROJ/${MODEL}.bin \
      -t x86_64-linux-clang \
      -l ${MODEL} \
      -o $PROJ \
      2>&1 | grep -E 'INFO|ERROR'
done

echo ""
echo "=== Results ==="
ls -lh $PROJ/generator_*.so $PROJ/libgenerator_*.so 2>/dev/null
echo "=== Done ==="
