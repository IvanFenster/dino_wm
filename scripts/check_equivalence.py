# [ARPL-WM] Equivalence checks for the opt-in speedups, run on a randomly initialized DINO-WM predictor.
#   1. ARPL_SDPA=1 (fused attention) gives the same output as the original attention, same weights.
#   2. The fused path keeps the frame-level causal mask: changing a later frame does not change earlier frames.
#   3. (GPU only) ARPL_AMP-style fp16 autocast gives the same loss as fp32 up to fp16 rounding.
# Usage: python scripts/check_equivalence.py [--device cpu|cuda]. Exits non-zero if a check fails.
import os, sys, argparse, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import vit

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()
dev = args.device

# Predictor shape as in PointMaze training: num_hist=3 frames x 196 patches, token width 384 + 10 + 10.
F, P, D = 3, 196, 404
_to = torch.Tensor.to
if dev == "cpu":  # models/vit.py moves the causal mask to 'cuda' when the model is built; keep it on CPU here
    torch.Tensor.to = lambda self, *a, **k: _to(self, *["cpu" if x == "cuda" else x for x in a], **k)
torch.manual_seed(0)
model = vit.ViTPredictor(num_patches=P, num_frames=F, dim=D, depth=6, heads=16, mlp_dim=2048,
                         dropout=0.1, emb_dropout=0, pool="mean").to(dev).eval()
torch.Tensor.to = _to
x = torch.randn(2, F * P, D, device=dev)

def run(sdpa, inp):
    os.environ["ARPL_SDPA"] = "1" if sdpa else "0"
    with torch.no_grad():
        return model(inp)

ok = True
ref, fused = run(False, x), run(True, x)
diff = (ref - fused).abs().max().item()
print(f"1. fused vs original attention, fp32: max abs diff {diff:.2e} (output scale {ref.abs().max().item():.2f})")
ok &= diff < 1e-4

x2 = x.clone(); x2[:, (F - 1) * P:] += torch.randn_like(x2[:, (F - 1) * P:])   # change only the last frame
out2 = run(True, x2)
leak = (out2[:, :(F - 1) * P] - fused[:, :(F - 1) * P]).abs().max().item()
moved = (out2[:, (F - 1) * P:] - fused[:, (F - 1) * P:]).abs().max().item()
print(f"2. causal mask, fused path: earlier frames changed by {leak:.2e}, last frame by {moved:.2e}")
ok &= leak < 1e-5 and moved > 1e-3

if dev == "cuda":
    tgt = torch.randn_like(x)
    def loss(amp, sdpa):
        os.environ["ARPL_SDPA"] = "1" if sdpa else "0"
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return torch.nn.functional.mse_loss(model(x).float(), tgt).item()
    l32, l16, l16s = loss(False, False), loss(True, False), loss(True, True)
    rel = max(abs(l16 - l32), abs(l16s - l32)) / l32
    print(f"3. loss fp32 {l32:.6f} | fp16 {l16:.6f} | fp16+fused {l16s:.6f} | max relative diff {rel:.1e}")
    ok &= rel < 1e-3
else:
    print("3. fp16 check skipped (needs a GPU)")

os.environ.pop("ARPL_SDPA", None)
print("ALL CHECKS PASSED" if ok else "CHECK FAILED")
sys.exit(0 if ok else 1)
