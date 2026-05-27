"""Extract .fmodel contents."""
import zipfile, os, sys

FMODEL = "/home/fibo/AI model/tts_models/fibotts_1.0.0_qcom_6490-8550_qnn_2.26_dsp_02c30b45759c7583f61881af2e1d8c24.fmodel"

try:
    with zipfile.ZipFile(FMODEL, 'r') as zf:
        print(f"Files in .fmodel ({len(zf.namelist())} entries):")
        for name in zf.namelist():
            info = zf.getinfo(name)
            print(f"  {name} ({info.file_size/1024/1024:.1f}MB)")

        # Extract decoder-related files
        out_dir = "/home/fibo/melotts_qnn/fmodel_extract"
        os.makedirs(out_dir, exist_ok=True)

        for name in zf.namelist():
            if 'decoder' in name.lower() or 'fibotts.bin' in name.lower() or 'config' in name.lower():
                zf.extract(name, out_dir)
                print(f"\nExtracted: {name}")
                # Show first 100 bytes
                with open(os.path.join(out_dir, name), 'rb') as f:
                    data = f.read(100)
                print(f"  Header: {data[:50].hex()}")

except zipfile.BadZipFile:
    print("Not a ZIP file")
    # Try other format detection
    with open(FMODEL, 'rb') as f:
        data = f.read(1024)
    print(f"First 1KB hex pattern:")
    # Look for ZIP signature, tar, etc
    for sig_name, sig in [("ZIP", b"PK"), ("GZIP", b"\x1f\x8b"), ("TAR", b"ustar")]:
        if sig in data:
            print(f"  Found {sig_name} signature at offset {data.index(sig)}")
