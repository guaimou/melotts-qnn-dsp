from fiboaisdk.api_aisdk_py import api_infer_py
api = api_infer_py.InferAPI()
methods = [m for m in dir(api) if "xecute" in m or "etch" in m]
for m in methods:
    print(m)
