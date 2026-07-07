"""
NICTO AI - Build C++ Engine

Usage:
    # Option 1: Using torch.utils.cpp_extension (recommended)
    python build_cpp.py

    # Option 2: Using setuptools directly
    python build_cpp.py --setuptools

    # Option 3: Manual
    cd nicto_ai/engine/cpp
    cmake -B build .
    cmake --build build --config Release
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

CPP_DIR = Path(__file__).parent / "cpp"
BUILD_DIR = CPP_DIR / "build"
ENGINE_DIR = Path(__file__).parent


def find_compiler():
    """Find available C++ compiler."""
    # Try MSVC
    if sys.platform == "win32":
        # Check for Visual Studio
        for vs_path in [
            os.environ.get("VCINSTALLDIR", ""),
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC",
            r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC",
            r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC",
        ]:
            if os.path.exists(vs_path):
                return "msvc"

    # Try g++
    for cmd in ["g++", "clang++", "c++"]:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True)
            return cmd
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    return None


def build_with_torch():
    """Build using torch.utils.cpp_extension (auto-handles compiler)."""
    print("Building with torch.utils.cpp_extension...")
    try:
        from torch.utils.cpp_extension import load
    except ImportError:
        print("ERROR: PyTorch not installed. Install with: pip install torch")
        return False

    sources = [str(CPP_DIR / "bindings.cpp")]

    extra_cflags = ["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3", "-std=c++17"]

    try:
        module = load(
            name="nicto_engine",
            sources=sources,
            extra_cflags=extra_cflags,
            verbose=True,
        )
        print(f"SUCCESS: Built {module.__file__}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def build_with_setuptools():
    """Build using setuptools."""
    print("Building with setuptools...")
    setup_py = CPP_DIR.parent / "setup_cpp.py"
    if not setup_py.exists():
        print(f"ERROR: {setup_py} not found")
        return False

    try:
        subprocess.run(
            [sys.executable, str(setup_py), "build_ext", "--inplace"],
            cwd=str(CPP_DIR.parent.parent),
            check=True,
        )
        print("SUCCESS: Built with setuptools")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        return False


def build_with_cmake():
    """Build using CMake."""
    print("Building with CMake...")
    BUILD_DIR.mkdir(exist_ok=True)

    try:
        subprocess.run(
            ["cmake", "-B", str(BUILD_DIR), "-S", str(CPP_DIR)],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(BUILD_DIR), "--config", "Release"],
            check=True,
        )
        print("SUCCESS: Built with CMake")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("  NICTO AI - C++ Engine Builder")
    print("=" * 60)

    compiler = find_compiler()
    print(f"Compiler: {compiler or 'NOT FOUND'}")

    if compiler is None:
        print()
        print("No C++ compiler found. Install one of:")
        print("  - Visual Studio Build Tools (Windows)")
        print("    https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        print("  - MinGW-w64 (Windows)")
        print("    https://www.mingw-w64.org/")
        print("  - g++ (Linux)")
        print("    sudo apt install g++")
        print("  - Xcode Command Line Tools (macOS)")
        print("    xcode-select --install")
        print()
        print("Python fallbacks will be used instead (still fast).")
        return False

    # Try torch.utils.cpp_extension first (easiest)
    if build_with_torch():
        return True

    # Try setuptools
    if build_with_setuptools():
        return True

    # Try CMake
    if build_with_cmake():
        return True

    print("All build methods failed. Python fallbacks will be used.")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
