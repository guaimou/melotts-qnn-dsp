"""Test QNN HTP backend (v68 native) - might support W8A16."""
import sys, time, numpy as np

sys.path.insert(0, "/usr/local/lib/python3.8/dist-packages")
from fiboaisdk.api_aisdk_py import api_infer_py
import onnxruntime as ort

SO = "/home/fibo/melotts_qnn/lib_int8_v2/aarch64-ubuntu-gcc9.4/libmelotts_int8_v2.so"
z = np.random.randn(1, 192, 128).astype(np.float32) * 0.5

# ONNX reference
sess = ort.InferenceSession("/home/fibo/melotts_qnn/generator_cle.onnx", providers=['CPUExecutionProvider'])
y_onnx = sess.run(None, {'z': z})[0].squeeze()

for dev in ["HTP", "DSP"]:
    print(f"\n=== QNN INT8 + {dev} ===")
    try:
        params = api_infer_py.InferParams(SO, 'QUALCOMM', 'QNN', dev, 'ERROR', 5)
        api = api_infer_py.InferAPI()
        api.Init(params)

        # INT8 encode
        z_nhwc = np.transpose(z, (0, 2, 1))
        z_u8 = np.clip(np.round(z_nhwc / 0.0636 + 133), 0, 255).astype(np.uint8)

        t0 = time.time()
        ret = api.Execute_uint8({"z": z_u8.flatten().tolist()})
        t_exec = time.time() - t0

        y_u8 = np.array(api.FetchOutputs_uint8(["y"])["y"], dtype=np.uint8)
        y_dsp = (y_u8.astype(np.float32) - 117) * 0.001663

        a = y_onnx[:len(y_dsp)]
        b = y_dsp[:len(a)]
        corr = np.corrcoef(a, b)[0,1]
        sp = np.var(a)
        npw = np.var(a-b)
        snr = 10*np.log10(sp/npw) if npw>0 else float('inf')

        print(f"  Exec: {t_exec*1000:.0f}ms, corr={corr:.6f}, SNR={snr:.1f}dB")
        api.Release()
    except Exception as e:
        print(f"  FAIL: {str(e)[:150]}")
