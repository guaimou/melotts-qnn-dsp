import sys
sys.path.insert(0, "/project/cosyvoice_snpe_snpe/qairt/lib/python")
from qti.aisw.converters.common.backend_awareness import BackendInfo

print("=== SDK 2.32.6 Supported SoCs (DSP) ===")
for soc in ["8550", "6490", "SM8550", "SM6490", "QCM6490", "8Gen2", "SA8295", "SA8255", "SM8450", "SM8350", "SM8650", "8Gen3"]:
    try:
        bi = BackendInfo.get_instance("DSP", soc)
        print(f"  {soc}: OK")
    except Exception as e:
        print(f"  {soc}: {str(e)[:80]}")
