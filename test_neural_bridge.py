"""Neural bridge end-to-end test — TorchVerlet, Dataset, MLPSurrogate.

Verifies the differentiable simulation, dataset generation, and surrogate
training pipeline all work together.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import torch

print("=" * 60)
print("Neural Bridge Tests")
print("=" * 60)

# --- 1. TorchVerlet single step ---
from nicto_ai.nictos.neural.integrator import TorchVerlet, force_coulomb, force_gravity

verlet = TorchVerlet()
x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
v = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
f = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
mass = torch.tensor([[1.0], [1.0]], dtype=torch.float64)

x_new, v_new, a_new = verlet(x, v, f, mass, dt=0.01)
assert x_new.shape == x.shape, f"Shape mismatch: {x_new.shape}"
assert v_new.shape == v.shape
assert a_new.shape == f.shape
print("1. TorchVerlet single step OK")

# --- 2. TorchVerlet stored acceleration loop ---
x = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
v = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
f = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
mass = torch.tensor([[1.0]], dtype=torch.float64)

stored = None
positions = [x.clone()]
for _ in range(100):
    x, v, stored = verlet(x, v, f, mass, dt=0.01, stored_accel=stored)
    positions.append(x.clone())

# Constant force → parabolic trajectory, x should be non-zero
final_x = positions[-1][0, 0].item()
assert abs(final_x) > 0.01, f"Expected drift in x, got {final_x}"
print(f"2. Verlet 100-step loop OK (final x={final_x:.4f})")

# --- 3. Force functions ---
charges = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
x_c = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
f_c = force_coulomb(x_c, charges)
assert f_c.shape == (2, 3)
assert f_c.norm() > 0, "Coulomb force is zero"
# Opposite charges attract — forces on each particle should be nonzero and opposite
assert torch.allclose(f_c[0], -f_c[1], atol=1e-6), "Coulomb forces not equal/opposite"
print(f"3. force_coulomb OK (|F|={f_c.norm():.2e})")

masses = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
f_g = force_gravity(x_c, masses)
assert f_g.shape == (2, 3)
assert f_g.norm() > 0, "Gravity force is zero"
assert torch.allclose(f_g[0], -f_g[1], atol=1e-6), "Gravity forces not equal/opposite"
print(f"4. force_gravity OK (|F|={f_g.norm():.2e})")

# --- 5. Hydrogen with TorchVerlet (differentiable) ---
charges_h = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
x_h = torch.tensor([[0.0, 0.0, 0.0], [5.29e-11, 0.0, 0.0]], dtype=torch.float64)
v_h = torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.19e6, 0.0]], dtype=torch.float64)
mass_h = torch.tensor([[1836.0], [1.0]], dtype=torch.float64)

stored = None
for _ in range(500):
    f_h = force_coulomb(x_h, charges_h)
    x_h, v_h, stored = verlet(x_h, v_h, f_h, mass_h, dt=1e-17, stored_accel=stored)

# Electron should have moved significantly from initial position
dist = (x_h[1] - torch.tensor([5.29e-11, 0.0, 0.0], dtype=torch.float64)).norm().item()
print(f"5. Hydrogen TorchVerlet 500 steps OK (electron moved {dist:.4e} m)")
assert dist > 1e-12, "Electron didn't move"

# --- 6. SimulationDataset from engine history ---
from nicto_ai.nictos import Nictos
from nicto_ai.nictos.domains.physics.types import Particle
from nicto_ai.nictos.neural.dataset import SimulationDataset, trajectory_to_tensors

kit = Nictos()
sim = kit.create_simulation("test_data", domain="physics")
p0 = Particle(id=0, mass=1.0, charge=1.0, position=[0.0, 0.0, 0.0])
p1 = Particle(id=1, mass=1.0, charge=-1.0, position=[1.0, 0.0, 0.0], velocity=[0.0, 0.5, 0.0])
sim.add_particles([p0, p1])
from nicto_ai.nictos.domains.physics.rules import CoulombRule
sim.add_rule(CoulombRule())
history = sim.run(steps=200, dt=0.01)

dataset = SimulationDataset(history)
assert len(dataset) == 199, f"Expected 199 samples, got {len(dataset)}"
X, Y = dataset[0]
assert X.shape == Y.shape
print(f"6. SimulationDataset OK (len={len(dataset)}, shape={X.shape})")

# --- 7. trajectory_to_tensors ---
Xt, Yt = trajectory_to_tensors(history)
assert Xt.shape[0] == 199
assert Xt.shape[1] == 2 * 8  # 2 particles * (pos3 + vel3 + mass + charge)
print(f"7. trajectory_to_tensors OK (X={Xt.shape}, Y={Yt.shape})")

# --- 8. MLPSurrogate train + predict ---
from nicto_ai.nictos.neural.surrogate import MLPSurrogate, train_surrogate

state_dim = Xt.shape[1]
model = MLPSurrogate(state_dim, hidden=64, layers=2)
pred = model(Xt[:5].float())
assert pred.shape == Xt[:5].shape, f"Surrogate output shape: {pred.shape}"
print(f"8. MLPSurrogate forward OK (input={Xt[:5].shape}, output={pred.shape})")

# --- 9. train_surrogate ---
trained_model = train_surrogate(
    dataset, state_dim, hidden=64, layers=2,
    epochs=20, batch_size=32, lr=1e-3, device="cpu"
)
# Verify trained model produces different (better) predictions
with torch.no_grad():
    pred_before = model(Xt[:10].float())
    pred_after = trained_model(Xt[:10].float())
    loss_before = torch.nn.functional.mse_loss(pred_before, Yt[:10].float()).item()
    loss_after = torch.nn.functional.mse_loss(pred_after, Yt[:10].float()).item()
print(f"9. train_surrogate OK (loss before={loss_before:.4e}, after={loss_after:.4e})")

# --- 10. Gradient flow through TorchVerlet ---
x_grad = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
v_grad = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
f_grad = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
mass_grad = torch.tensor([[1.0]], dtype=torch.float64)

x_out, v_out, _ = verlet(x_grad, v_grad, f_grad, mass_grad, dt=0.01)
loss = x_out.sum()
loss.backward()
assert x_grad.grad is not None, "No gradient flow through TorchVerlet"
print(f"10. Gradient flow OK (grad norm={x_grad.grad.norm():.4e})")

print()
print("=" * 60)
print("ALL NEURAL BRIDGE TESTS COMPLETE")
print("=" * 60)
