import sys, torch
sys.stdout.reconfigure(encoding='utf-8')
from nicto_ai.training.model_master import config_master_tiny, NICTOMasterModel

cfg = config_master_tiny()
model = NICTOMasterModel(cfg)
ckpt = torch.load('checkpoints_master/tiny/final.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'])
torch.save({'model': model.state_dict(), 'config': cfg, 'step': 0}, 'checkpoints_master/tiny/restart.pt')
print(f'Saved restart checkpoint from final.pt (step={ckpt.get("step", "?")})')
