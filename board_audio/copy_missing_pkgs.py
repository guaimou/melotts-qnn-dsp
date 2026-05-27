"""Find all packages in system Python site-packages that melo might need."""
import os, shutil, sys

system_sp = '/usr/local/lib/python3.8/dist-packages'
conda_sp = '/home/fibo/qwen3_tts/.conda_env/lib/python3.12/site-packages'

# Packages known to be needed by melo
needed = ['txtsplit', 'cn2an', 'jieba', 'pypinyin', 'opencc']

for pkg in needed:
    src = os.path.join(system_sp, pkg)
    dst = os.path.join(conda_sp, pkg)
    if os.path.exists(src) and not os.path.exists(dst):
        print(f"Copying {pkg}...")
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    elif os.path.exists(dst):
        print(f"{pkg}: already in conda")
    else:
        print(f"{pkg}: not found in system either")
