#!/bin/bash
SDK=/opt/qairt/2.26.2.240911
export LD_LIBRARY_PATH=$SDK/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
PROJ=/project/cosyvoice_snpe_snpe

echo "z:=$PROJ/z_input.raw" > $PROJ/input262.txt

echo "=== QNN CPU Inference ==="
$SDK/bin/x86_64-linux-clang/qnn-net-run \
  --model $PROJ/lib262/x86_64-linux-clang/libmelotts_gen.so \
  --backend $SDK/lib/x86_64-linux-clang/libQnnCpu.so \
  --input_list $PROJ/input262.txt \
  --output_dir $PROJ/output262 \
  2>&1 | tail -5

echo ""
echo "=== Output ==="
python3 << 'PYEOF'
import numpy as np, os
proj = "/project/cosyvoice_snpe_snpe"
for root, dirs, files in os.walk(proj + "/output262"):
    for f in files:
        if f.endswith('.raw'):
            fpath = os.path.join(root, f)
            data = np.fromfile(fpath, dtype=np.float32)
            print("  %s: shape %s, mean=%.4f, std=%.4f" % (f, str(data.shape), data.mean(), data.std()))
PYEOF
