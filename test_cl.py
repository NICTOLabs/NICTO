import subprocess, sys, os

vsdevcmd = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"

# Run cl.exe through VsDevCmd.bat to set up environment
cmd = f'"{vsdevcmd}" -test && cl /?'
r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True, timeout=30)
print("STDOUT:", r.stdout[:500])
print("STDERR:", r.stderr[:500])
print("RC:", r.returncode)
