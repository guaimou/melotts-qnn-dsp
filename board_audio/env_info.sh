#!/bin/bash
echo "=== Board System ==="
uname -a
head -3 /etc/os-release 2>/dev/null
echo ""
echo "=== CPU ==="
cat /proc/cpuinfo 2>/dev/null | grep -m1 "model name\|Processor\|Hardware"
echo ""
echo "=== DSP/SoC ==="
cat /sys/devices/soc0/soc_id 2>/dev/null
cat /sys/devices/soc0/machine 2>/dev/null
echo ""
echo "=== Python ==="
python3 --version 2>/dev/null
python3.8 --version 2>/dev/null
echo ""
echo "=== Key packages (system python3) ==="
python3 -c "import torch; print('torch:', torch.__version__)" 2>/dev/null
python3 -c "import numpy; print('numpy:', numpy.__version__)" 2>/dev/null
python3 -c "import onnxruntime; print('onnxruntime:', onnxruntime.__version__)" 2>/dev/null
python3 -c "import scipy; print('scipy:', scipy.__version__)" 2>/dev/null
echo ""
echo "=== fiboaisdk ==="
pip3 list 2>/dev/null | grep -iE "fibo|qnn|snpe|onnx"
python3 -c "import fiboaisdk; print('fiboaisdk imported')" 2>/dev/null || echo "fiboaisdk not directly importable"
echo ""
echo "=== QNN SDK ==="
ls /home/fibo/qnn_sdk_262/ 2>/dev/null | head -5
echo ""
echo "=== GCC ==="
g++ --version 2>/dev/null | head -1
aarch64-linux-gnu-g++ --version 2>/dev/null | head -1
echo ""
echo "=== License ==="
ls /home/fibo/qcom_6490_license/ 2>/dev/null
echo ""
echo "=== TTS model ==="
ls -lh "/home/fibo/TTS/" 2>/dev/null | head -5
echo ""
echo "=== fiboaisdk path ==="
pip3 show fiboaisdk 2>/dev/null | grep -E "Version|Location"
