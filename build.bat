@echo off
setlocal

set VCVARSALL=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat
set MSVC_BIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
set PYTHON_INCLUDE=C:\Users\BYU\AppData\Local\Programs\Python\Python312\Include
set PYBIND_INCLUDE=C:\Users\BYU\AppData\Local\Programs\Python\Python312\Lib\site-packages\pybind11\include
set UCRT_INCLUDE=C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt
set UM_INCLUDE=C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um
set SHARED_INCLUDE=C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared
set UCRT_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64
set UM_LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64
set PYTHON_LIB=C:\Users\BYU\AppData\Local\Programs\Python\Python312\libs\python312.lib

call "%VCVARSALL%" x64 >nul 2>&1
set PATH=%MSVC_BIN%;%PATH%

cd /d C:\Users\BYU\Desktop\NICTO

echo.
echo === Building nicto_engine.pyd ===
cl /O2 /std:c++17 /EHsc /LD ^
  /I"%PYTHON_INCLUDE%" ^
  /I"%PYBIND_INCLUDE%" ^
  /I"%UCRT_INCLUDE%" ^
  /I"%UM_INCLUDE%" ^
  /I"%SHARED_INCLUDE%" ^
  /I"nicto_ai\engine\cpp" ^
  nicto_ai\engine\cpp\bindings.cpp ^
  /Fe:nicto_ai\engine\nicto_engine.pyd ^
  /link /DLL "%PYTHON_LIB%" /IMPLIB:nicto_ai\engine\nicto_engine.lib
if errorlevel 1 goto :fail

echo.
echo === BUILD SUCCESS ===

echo.
echo === Testing ===
"C:\Users\BYU\AppData\Local\Programs\Python\Python312\python.exe" -u -c "import sys; sys.path.insert(0,'.'); from nicto_ai.engine import nicto_engine; print('C++ ENGINE LOADED!'); r=nicto_engine.encode('Hello NICTO!',32000); print('encode:',r); print('decode:',nicto_engine.decode(r)); print('sample:',nicto_engine.sample([0.1]*32000,32000,0.8,50)); print('find_stop:',nicto_engine.find_stop('Hello\n\nAssistant: Hi'))"
goto :end

:fail
echo.
echo === BUILD FAILED ===

:end
