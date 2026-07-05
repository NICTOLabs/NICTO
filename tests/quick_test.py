"""Quick smoke test - CPU mode with small model"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys, torch
sys.path.insert(0, r"C:\Users\BYU\Desktop\NICTO")

print(f"PyTorch: {torch.__version__}")

from nicto_ai.core.model import create_small_model
from nicto_ai.core.vision import VGG16ForNICTO

print("\n[1/5] Creating small NICTO model...")
model = create_small_model(vocab_size=32000, dim=256, max_seq_len=512)
model.vgg16 = VGG16ForNICTO(nicto_dim=256, pretrained=False)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params:,}")

print("\n[2/5] Forward pass...")
input_ids = torch.randint(0, 32000, (2, 16))
outputs = model(input_ids)
print(f"  Input:  {input_ids.shape}")
print(f"  Output: {outputs['logits'].shape}")

print("\n[3/5] Loss computation...")
labels = torch.randint(0, 32000, (2, 16))
outputs = model(input_ids, labels=labels)
print(f"  Loss: {outputs['loss'].item():.4f}")

print("\n[4/5] Vision encoding...")
images = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    vis = model.vgg16(images)
print(f"  Global: {vis['global_features'].shape}")
print(f"  Patches: {vis['patch_features'].shape}")

print("\n[5/5] Text generation...")
prompt = torch.randint(0, 32000, (1, 5))
with torch.no_grad():
    gen = model.generate(prompt, max_new_tokens=5, temperature=1.0)
print(f"  Prompt: {prompt.shape}")
print(f"  Generated: {gen.shape}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED - NICTO AI is ready to train!")
print("=" * 50)
