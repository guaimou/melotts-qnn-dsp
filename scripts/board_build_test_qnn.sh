#!/bin/bash
set -e
cd /home/fibo/melotts_qnn
SDK=/home/fibo/qnn_sdk_262

echo "=== Build INT16 .so ==="
QNN_AARCH64_UBUNTU_GCC_94='' PYTHONPATH=$SDK/lib/python:$PYTHONPATH \
python3 $SDK/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c gen262_int16.cpp -b gen262_int16.bin \
  -t aarch64-ubuntu-gcc9.4 -l melotts_int16 -o ./lib_int16 \
  2>&1 | tail -3

INT16_SO=lib_int16/aarch64-ubuntu-gcc9.4/libmelotts_int16.so
ls -lh $INT16_SO

echo ""
echo "=== Build INT8 .so ==="
QNN_AARCH64_UBUNTU_GCC_94='' PYTHONPATH=$SDK/lib/python:$PYTHONPATH \
python3 $SDK/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c gen262_int8.cpp -b gen262_int8.bin \
  -t aarch64-ubuntu-gcc9.4 -l melotts_int8 -o ./lib_int8 \
  2>&1 | tail -3

INT8_SO=lib_int8/aarch64-ubuntu-gcc9.4/libmelotts_int8.so
ls -lh $INT8_SO

echo ""
echo "=== Test QNN DSP INT16 ==="
python3 -c "
import time, numpy as np
from fiboaisdk.api_aisdk_py import api_infer_py

z = np.random.randn(1, 192, 128).astype(np.float32)
feed = {'z': z.flatten().tolist()}
so = '$INT16_SO'

print(f'Model: {so}')
params = api_infer_py.InferParams(so, 'QUALCOMM', 'QNN', 'DSP', 'ERROR', 5)
api = api_infer_py.InferAPI()
ret = api.Init(params)
if ret != 0:
    print(f'Init FAILED: {ret}')
else:
    ret = api.Execute_float(feed)
    if ret != 0:
        print(f'Execute FAILED: {ret}')
    else:
        times = []
        for i in range(3):
            t0 = time.perf_counter()
            ret = api.Execute_float(feed)
            if ret != 0: break
            times.append(time.perf_counter() - t0)
        if times:
            avg_ms = np.mean(times) * 1000
            rtf = (avg_ms/1000) / (65536/44100)
            print(f'Avg: {avg_ms:.0f}ms | RTF: {rtf:.3f}')
    api.Release()
" 2>&1

echo ""
echo "=== Test QNN DSP INT8 ==="
python3 -c "
import time, numpy as np
from fiboaisdk.api_aisdk_py import api_infer_py

z = np.random.randn(1, 192, 128).astype(np.float32)
feed = {'z': z.flatten().tolist()}
so = '$INT8_SO'

print(f'Model: {so}')
params = api_infer_py.InferParams(so, 'QUALCOMM', 'QNN', 'DSP', 'ERROR', 5)
api = api_infer_py.InferAPI()
ret = api.Init(params)
if ret != 0:
    print(f'Init FAILED: {ret}')
else:
    ret = api.Execute_float(feed)
    if ret != 0:
        print(f'Execute FAILED: {ret}')
    else:
        times = []
        for i in range(3):
            t0 = time.perf_counter()
            ret = api.Execute_float(feed)
            if ret != 0: break
            times.append(time.perf_counter() - t0)
        if times:
            avg_ms = np.mean(times) * 1000
            rtf = (avg_ms/1000) / (65536/44100)
            print(f'Avg: {avg_ms:.0f}ms | RTF: {rtf:.3f}')
    api.Release()
" 2>&1

echo ""
echo "=== Done ==="
