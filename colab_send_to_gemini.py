"""
NICTO AI - Send code to Gemini for training assistance
Run this in Colab to let Gemini analyze and help train your model.
"""

import os
import glob

# ============================================================
# CONFIG - Set your Gemini API key
# ============================================================
# Option 1: Use Colab Secrets (recommended)
# from google.colab import userdata
# API_KEY = userdata.get('GOOGLE_API_KEY')

# Option 2: Hardcode (NOT recommended for sharing)
API_KEY = "YOUR_API_KEY_HERE"

# ============================================================
# Load all NICTO Python files
# ============================================================
NICTO_DIR = "nicto_ai"  # Adjust path if needed

def load_codebase(directory):
    """Load all Python files from the NICTO codebase."""
    code_files = {}
    for filepath in sorted(glob.glob(os.path.join(directory, "**", "*.py"), recursive=True)):
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            # Skip huge files and __pycache__
            if "__pycache__" in filepath:
                continue
            if len(content) > 50000:  # Skip files > 50KB
                print(f"  Skipping (too large): {filepath}")
                continue
            code_files[filepath] = content
            print(f"  Loaded: {filepath} ({len(content)} chars)")
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
    return code_files

print("Loading NICTO codebase...")
codebase = load_codebase(NICTO_DIR)
print(f"\nLoaded {len(codebase)} files")
total_chars = sum(len(v) for v in codebase.values())
print(f"Total size: {total_chars:,} chars (~{total_chars // 4:,} tokens)")

# ============================================================
# Send to Gemini
# ============================================================
import google.generativeai as genai

genai.configure(api_key=API_KEY)

# Use gemini-2.5-flash (free tier, 1M context window)
model = genai.GenerativeModel("gemini-2.5-flash")

# Build the prompt
code_context = "\n\n".join([
    f"=== FILE: {filepath} ===\n```python\n{content}\n```"
    for filepath, content in codebase.items()
])

prompt = f"""You are analyzing the NICTO AI codebase. This is a neural network architecture with:
- Multi-Latent Attention (MLA)
- Mixture of Experts (MoE) 
- Mamba SSM
- Liquid Neural Networks
- Consciousness/Metacognition layers
- Emotion system
- Hierarchical memory
- Fusion gate

The codebase has {len(codebase)} Python files ({total_chars:,} chars total).

Here is the full codebase:

{code_context}

TASK: 
1. Verify all components actually work (no mocks, no stubs)
2. Identify any bugs or issues
3. Suggest how to train this model on a T4 GPU (16GB VRAM)
4. What's the maximum model size that fits on T4?
5. What training data should I use?
6. Write a complete training script that works on Colab Free tier

Be specific and give me working code I can run."""

print("\nSending to Gemini...")
print("(This may take 30-60 seconds for a large codebase)\n")

response = model.generate_content(prompt)

print("=" * 60)
print("GEMINI'S ANALYSIS:")
print("=" * 60)
print(response.text)

# Save response
with open("gemini_analysis.md", "w", encoding="utf-8") as f:
    f.write("# NICTO Codebase Analysis by Gemini\n\n")
    f.write(response.text)
print(f"\nSaved to gemini_analysis.md")
