#!/usr/bin/env python3
"""ROIJoy diagnostic script — run this to troubleshoot installation issues.

Usage:
    cd ROIJoy
    python debug.py

Share the full output with ChatGPT or your advisor to get help.
"""
import sys
import os
import platform
import shutil
import subprocess

WIDTH = 60
def header(title):
    print(f"\n{'=' * WIDTH}")
    print(f"  {title}")
    print(f"{'=' * WIDTH}")

def check(label, ok, detail=""):
    status = "OK" if ok else "FAIL"
    symbol = "+" if ok else "X"
    print(f"  [{symbol}] {label}: {status}")
    if detail:
        print(f"      {detail}")
    return ok

all_ok = True

header("SYSTEM INFO")
print(f"  Python:   {sys.version}")
print(f"  Platform: {platform.platform()}")
print(f"  OS:       {platform.system()} {platform.release()}")
print(f"  CWD:      {os.getcwd()}")
print(f"  Script:   {os.path.abspath(__file__)}")

header("PROJECT STRUCTURE")
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
print(f"  Project root: {project_root}")

required_files = [
    "roijoy/__init__.py",
    "roijoy/app.py",
    "roijoy/layout.py",
    "roijoy/callbacks.py",
    "roijoy/envi_io.py",
    "roijoy/roi.py",
    "roijoy/matching.py",
    "roijoy/assets/style.css",
    "requirements.txt",
]

for f in required_files:
    exists = os.path.exists(os.path.join(project_root, f))
    if not check(f, exists):
        all_ok = False

header("DATA FILES")
data_dir = os.path.join(project_root, "data")
if os.path.isdir(data_dir):
    hdr_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".hdr")])
    bin_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".bin")])
    print(f"  .hdr files: {len(hdr_files)}")
    print(f"  .bin files: {len(bin_files)}")

    for hdr in hdr_files:
        base = hdr.replace(".hdr", "")
        bin_name = base + ".bin"
        bin_path = os.path.join(data_dir, bin_name)
        has_bin = os.path.exists(bin_path) and not os.path.islink(bin_path)
        is_broken_link = os.path.islink(bin_path) and not os.path.exists(bin_path)

        if has_bin:
            size_mb = os.path.getsize(bin_path) / 1e6
            check(base, True, f".bin present ({size_mb:.1f} MB)")
        elif is_broken_link:
            target = os.readlink(bin_path)
            check(base, False, f".bin is a broken symlink -> {target}")
            all_ok = False
        elif os.path.islink(bin_path):
            size_mb = os.path.getsize(bin_path) / 1e6
            check(base, True, f".bin is a symlink ({size_mb:.1f} MB, resolves OK)")
        else:
            check(base, False, f".bin MISSING — needed to load this image")
            all_ok = False
else:
    check("data/ directory", False, "directory not found")
    all_ok = False

header("PYTHON DEPENDENCIES")
required_packages = [
    ("dash", "dash"),
    ("flask", "Flask"),
    ("plotly", "plotly"),
    ("numpy", "numpy"),
    ("spectral", "spectral"),
    ("skimage", "scikit-image"),
    ("cv2", "opencv-python"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("PIL", "Pillow"),
]

for import_name, pip_name in required_packages:
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "?")
        check(pip_name, True, f"v{version}")
    except ImportError:
        check(pip_name, False, f"pip install {pip_name}")
        all_ok = False

header("VIRTUAL ENVIRONMENT")
in_venv = sys.prefix != sys.base_prefix
check("Running inside venv", in_venv,
      f"prefix={sys.prefix}" if in_venv else
      "Run: source .venv/bin/activate (macOS/Linux) or .venv\\Scripts\\activate (Windows)")
if not in_venv:
    all_ok = False

venv_dir = os.path.join(project_root, ".venv")
check(".venv/ directory exists", os.path.isdir(venv_dir),
      "" if os.path.isdir(venv_dir) else "Run: python3 -m venv .venv")

header("PORT AVAILABILITY")
import socket
for port in [8050, 8051, 8052]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(("127.0.0.1", port))
    sock.close()
    if result != 0:
        check(f"Port {port}", True, "available")
        break
    else:
        check(f"Port {port}", False, "IN USE — something is already running on this port")

header("ENVI FILE LOADING TEST")
try:
    from roijoy.envi_io import load_envi
    test_hdrs = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".hdr")
    ]) if os.path.isdir(data_dir) else []

    if test_hdrs:
        hdr = test_hdrs[0]
        cube, wavelengths = load_envi(hdr)
        check("Load ENVI cube", True,
              f"shape={cube.shape}, wavelengths={len(wavelengths)} "
              f"({wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm)")
    else:
        check("Load ENVI cube", False, "No .hdr files found in data/")
        all_ok = False
except Exception as e:
    check("Load ENVI cube", False, str(e))
    all_ok = False

header("DASH APP IMPORT TEST")
try:
    from roijoy.layout import create_layout
    layout = create_layout()
    check("Import layout", True)
except Exception as e:
    check("Import layout", False, str(e))
    all_ok = False

try:
    from roijoy import callbacks
    check("Import callbacks", True)
except Exception as e:
    check("Import callbacks", False, str(e))
    all_ok = False

header("BROWSER")
import webbrowser
browser = webbrowser.get()
browser_name = type(browser).__name__ if browser else "None"
check("Default browser detected", browser is not None, browser_name)
print(f"\n  TIP: If the browser doesn't open automatically, manually go to:")
print(f"       http://localhost:8050")
print(f"       (or http://127.0.0.1:8050)")

header("SUMMARY")
if all_ok:
    print("  All checks passed! ROIJoy should work.")
    print(f"\n  To start:")
    print(f"    cd {project_root}")
    if not in_venv:
        print(f"    source .venv/bin/activate")
    print(f"    python -m roijoy.app")
    print(f"    Then open http://localhost:8050 in your browser")
else:
    print("  Some checks FAILED — see [X] items above.")
    print("  Fix those issues and run this script again.")
    print(f"\n  Quick fix attempt:")
    print(f"    cd {project_root}")
    print(f"    python3 -m venv .venv")
    if platform.system() == "Windows":
        print(f"    .venv\\Scripts\\activate")
    else:
        print(f"    source .venv/bin/activate")
    print(f"    pip install -r requirements.txt")
    print(f"    python debug.py")

print()
