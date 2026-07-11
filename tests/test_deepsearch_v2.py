import torch
from nicto_ai.core.deepsearch import DeepSearchModule

print("Testing DeepSearchModule v2...")
model = DeepSearchModule(dim=256, max_depth=5, beam_width=2, n_thoughts_per_step=3)

# Test forward pass
x = torch.randn(2, 256)
result = model(x, use_search=False, n_thoughts=3)
print("Training mode output shape:", result["best_reasoning"].shape)
print("Best score:", result["best_score"].mean().item())
print("Depth reached:", result["depth_reached"])

# Test beam search
model.eval()
result = model(x, use_search=True, n_thoughts=3)
print("Beam search output shape:", result["best_reasoning"].shape)
print("Best score:", result["best_score"].mean().item())
print("Depth reached:", result["depth_reached"])
print("Search history:", len(result["all_scores"]), "steps")

# Count parameters
params = sum(p.numel() for p in model.parameters())
print("DeepSearch v2 params:", f"{params:,}", f"({params*4/1024/1024:.1f} MB)")

print("All DeepSearch v2 tests passed!")
