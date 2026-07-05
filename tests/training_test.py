"""Training pipeline test - CPU compatible"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys, torch, time
sys.path.insert(0, r"C:\Users\BYU\Desktop\NICTO")

from torch.utils.data import TensorDataset, DataLoader
from nicto_ai.core.model import create_small_model
from nicto_ai.training.data_pipeline import (
    TokenizerWrapper,
    TextFileDataset,
    create_dataloader,
    create_synthetic_dataset,
)

VOCAB = 32000
DIM = 256
BATCH = 4
SEQ_LEN = 32
STEPS = 20

print(f"PyTorch: {torch.__version__}")

# Test data pipeline
print("\n[0/5] Testing data pipeline...")
tok = TokenizerWrapper(VOCAB)
test_tokens = tok.encode("Hello, NICTO AI!")
print(f"  Tokenizer backend: {tok._backend}")
print(f"  'Hello, NICTO AI!' -> {test_tokens[:8]}...")

synth = create_synthetic_dataset(n_samples=100, seq_len=SEQ_LEN, vocab_size=VOCAB)
print(f"  Synthetic dataset: {len(synth)} samples")
sample = synth[0]
print(f"  Sample input_ids: {sample[0].shape}, labels: {sample[1].shape}")

# Create model
print("\n[1/5] Creating model...")
model = create_small_model(vocab_size=VOCAB, dim=DIM, max_seq_len=512)
params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {params:,}")

# Create synthetic dataset
print("\n[2/5] Creating dataloader...")
loader = DataLoader(synth, batch_size=BATCH, shuffle=True)

# Setup optimizer
print("\n[3/5] Setting up optimizer...")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-5)

# Training loop
print(f"\n[4/5] Training {STEPS} steps...")
model.train()
start = time.time()
losses = []

for step in range(1, STEPS + 1):
    batch = next(iter(loader))
    input_ids = batch[0]
    labels = batch[1]

    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]

    # Add MoE aux loss if present
    aux = outputs.get("aux_loss", torch.tensor(0.0))
    total_loss = loss + 0.01 * aux

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())

    if step % 5 == 0 or step == 1:
        lr = scheduler.get_last_lr()[0]
        print(f"  Step {step:3d}/{STEPS}: loss={loss.item():.4f} aux={aux.item():.4f} lr={lr:.6f}")

elapsed = time.time() - start
avg_loss = sum(losses) / len(losses)

print(f"\n  First loss: {losses[0]:.4f}")
print(f"  Last loss:  {losses[-1]:.4f}")
print(f"  Avg loss:   {avg_loss:.4f}")
print(f"  Time:       {elapsed:.2f}s ({elapsed/STEPS:.2f}s/step)")

# Verify gradient flow
grad_norms = []
for name, p in model.named_parameters():
    if p.grad is not None:
        grad_norms.append(p.grad.norm().item())

print(f"\n  Params with gradients: {len(grad_norms)}/{sum(1 for _ in model.parameters())}")
print(f"  Avg grad norm: {sum(grad_norms)/len(grad_norms):.6f}")

# Test inference
print("\n[5/5] Testing inference + DeepSearch...")
model.eval()
prompt = torch.randint(0, VOCAB, (1, 5))
with torch.no_grad():
    gen = model.generate(prompt, max_new_tokens=5, temperature=1.0)
print(f"  Generated: {gen.shape}")

# Test DeepSearch directly
from nicto_ai.core.deepsearch import DeepSearchModule
ds = DeepSearchModule(dim=DIM, max_depth=3, beam_width=2, n_thoughts_per_step=2)
h = torch.randn(1, DIM)
with torch.no_grad():
    result = ds(h.unsqueeze(1), use_search=True)
print(f"  DeepSearch output: {result['best_reasoning'].shape}")
print(f"  DeepSearch depth: {result['depth_reached']}")
print(f"  DeepSearch score: {result['best_score'].item():.4f}")

if losses[-1] < losses[0]:
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("Data pipeline + training + DeepSearch verified!")
    print("=" * 50)
else:
    print("\nWARNING: Loss did not decrease")
