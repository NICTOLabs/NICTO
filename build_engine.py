"""Build NICTO C++ engine using vcvarsall.bat environment"""

import os, sys, subprocess

# Find vcvarsall.bat
vcvarsall = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if not os.path.exists(vcvarsall):
    print("ERROR: vcvarsall.bat not found at", vcvarsall)
    sys.exit(1)

print("Found vcvarsall.bat:", vcvarsall)

# Use vcvarsall.bat to set up environment, then run cl
# This creates a proper shell with all VS environment variables
cmd = f'"{vcvarsall}" x64 && cl /?'
r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True, timeout=30)
if "Microsoft" in r.stderr:
    print("cl.exe: WORKS")
else:
    print("cl.exe: FAILED")
    print(r.stderr[:300])
    sys.exit(1)

# Now build - use a bat script to run the full build
build_script = r"""
@"%~dp0vcvarsall.bat" x64 >nul 2>&1
"""
# Write a build batch file
bat_content = f'''@echo off
call "{vcvarsall}" x64 >nul 2>&1
echo Environment set up. Running build...
cd /d "{os.getcwd()}"
"{sys.executable}" -c "import torch; from torch.utils.cpp_extension import load; import os,sys; os.environ['PATH']=r'{msvc_bin};'+os.environ.get('PATH',''); module=load(name='nicto_engine',sources=[r'{os.path.join('nicto_ai','engine','cpp','bindings.cpp')}'],extra_cflags=['/O2','/std:c++17'],verbose=True); print('BUILT:', module.__file__)"
'''

msvc_bin = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
bindings_cpp = os.path.join("nicto_ai", "engine", "cpp", "bindings.cpp")

bat_content = f'''@echo off
call "{vcvarsall}" x64 >nul 2>&1
set PATH={msvc_bin};%PATH%
cd /d "{os.getcwd()}"
"{sys.executable}" -u -c "import torch; from torch.utils.cpp_extension import load; module=load(name='nicto_engine',sources=[r'{bindings_cpp}'],extra_cflags=['/O2','/std:c++17'],verbose=True); print('BUILT:', module.__file__); r=module.encode('Hello NICTO!',32000); print('encode:',r); print('decode:',module.decode(r)); print('sample:',module.sample([0.1]*32000,32000,0.8,50))"
'''

bat_path = os.path.join(os.getcwd(), "build.bat")
with open(bat_path, "w") as f:
    f.write(bat_content)

print("Running build...")
r = subprocess.run(["cmd", "/c", bat_path], timeout=300)
print("Exit code:", r.returncode)
