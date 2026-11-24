# [ARPL-WM] Depth pilot: does adding DeFM depth tokens help DINO-WM's predictor on Granular?
# Moved from Depth_addition.ipynb (cells 7 to 11), logic unchanged. Steps:
#   1. Encode every frame once with the frozen encoders (DINOv2 for RGB, DeFM for depth, both at 196x196 -> 196 tokens).
#   2. Train DINO-WM's ViTPredictor on the cached tokens for 3 variants: rgb (384), depth (384), concat (768 per patch).
#   3. Save curves.png and logs.json. Metric "relative" = val MSE / MSE of guessing "next frame = current frame".
# Settings follow the paper for Rope/Granular (Table 11): H=1, frameskip=1, predictor lr 5e-5, encoder lr 5e-4, batch 32.
# Data: <data_root>/deformable/granular with actions.pth and per-episode obses.pth + 00.h5..19.h5 (scripts/get_deformable.sh).
# Needs the DeFM repo on PYTHONPATH (git clone https://github.com/leggedrobotics/defm) and a GPU.
import os, sys, json, time, argparse
import numpy as np, h5py, torch, torch.nn.functional as F
import torchvision.transforms as T

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from models.vit import ViTPredictor                    # DINO-WM predictor (unchanged)
from models.proprio import ProprioceptiveEmbedding     # DINO-WM action / proprio encoders
from defm import preprocess_depth_image                # DeFM 3-channel metric depth format
from defm.model_factory import create_defm_model

p = argparse.ArgumentParser()
p.add_argument("--data_root", default=os.environ.get("DATASET_DIR", "/content/data"))
p.add_argument("--n_ep", type=int, default=200)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--variants", default="rgb,depth,concat")
p.add_argument("--out", default="depth_pilot_out")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
N_EP, dev = args.n_ep, "cuda"
os.makedirs(args.out, exist_ok=True)
print(torch.__version__, torch.cuda.get_device_name(0))

# 1) Encode every frame ONCE with the frozen encoders, cache tokens on the GPU
root = f"{args.data_root}/deformable/granular"
actions = torch.load(f"{root}/actions.pth").float()[:N_EP]                        # (N, T, 4)
print("actions", tuple(actions.shape))
# Pinned DINOv2 commit, same as models/dino.py (the notebook used the latest, which needs Python >= 3.10)
dino = torch.hub.load("facebookresearch/dinov2:85a24602099d397264d5b30461ad7f3bfd726ca1", "dinov2_vits14",
                      trust_repo=True, skip_validation=True).to(dev).eval().half()
defm = create_defm_model("defm_vit_s14", pretrained=True).to(dev).eval().half()
rgb_tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.Normalize([0.5] * 3, [0.5] * 3)])  # DINO-WM data loader
to196 = T.Resize(196, antialias=True)                                                    # DINO-WM model resize, widened to DeFM

@torch.no_grad()
def enc(model, x):
    return model.forward_features(x.half())["x_norm_patchtokens"]                       # (B, 196, 384)

feats_rgb, feats_dep, t0 = [], [], time.time()
for i in range(N_EP):
    ep = f"{root}/{i:06d}"
    rgb = torch.load(f"{ep}/obses.pth").permute(0, 3, 1, 2) / 255.0                   # (20, 3, 224, 224)
    rgb = to196(rgb_tf(rgb)).to(dev)
    dep_m = [h5py.File(f"{ep}/{t:02d}.h5", "r")["observations/depth/cam_1"][0] / 10000.0 for t in range(20)]  # meters
    dep = torch.cat([preprocess_depth_image(d, target_size=224) for d in dep_m])       # (20, 3, 224, 224) DeFM format
    dep = to196(dep).to(dev)
    feats_rgb.append(enc(dino, rgb)); feats_dep.append(enc(defm, dep))
    if i == 0: print("per frame:", tuple(rgb.shape[1:]), "->", tuple(feats_rgb[0].shape[1:]), "| depth", tuple(dep.shape[1:]), "->", tuple(feats_dep[0].shape[1:]))
feats_rgb, feats_dep = torch.stack(feats_rgb), torch.stack(feats_dep)                  # (N, 20, 196, 384)
print("RGB tokens", tuple(feats_rgb.shape), "| depth tokens", tuple(feats_dep.shape), f"| {time.time() - t0:.0f}s")
s_rgb, s_dep = feats_rgb.float().std().item(), feats_dep.float().std().item()
print(f"feature std: DINOv2 {s_rgb:.3f} | DeFM {s_dep:.3f} | ratio {s_dep / s_rgb:.2f}")

# 2) Train DINO-WM's predictor on the cached tokens. concat_dim=1, normalize_action=True, split_ratio 0.9
a_flat = actions.reshape(-1, 4)
acts = ((actions - a_flat.mean(0)) / (a_flat.std(0) + 1e-6)).to(dev)
n_tr = int(0.9 * N_EP)
scale_dep = s_rgb / s_dep                     # fixed constant: puts the DeFM half on DINOv2's scale (nothing trained)
T_ACT = acts.shape[1]

def visual(variant, ep, t):
    if variant == "rgb": return feats_rgb[ep, t].float()
    if variant == "depth": return feats_dep[ep, t].float()
    return torch.cat([feats_rgb[ep, t].float(), feats_dep[ep, t].float() * scale_dep], -1)  # (B, 196, 768)

def make_pairs(eps): return torch.tensor([(e, t) for e in eps for t in range(19)])
tr_pairs, va_pairs = make_pairs(range(n_tr)), make_pairs(range(n_tr, N_EP))

def run(variant, epochs, bs):
    D = 768 if variant == "concat" else 384
    torch.manual_seed(args.seed)
    pred = ViTPredictor(num_patches=196, num_frames=1, dim=D + 20, depth=6, heads=16, mlp_dim=2048,
                        dropout=0.1, emb_dropout=0, pool="mean").to(dev)
    act_enc = ProprioceptiveEmbedding(num_frames=1, tubelet_size=1, in_chans=4, emb_dim=10).to(dev)
    prop_enc = ProprioceptiveEmbedding(num_frames=1, tubelet_size=1, in_chans=1, emb_dim=10).to(dev)
    opt_p = torch.optim.AdamW(pred.parameters(), lr=5e-5)
    opt_a = torch.optim.AdamW(list(act_enc.parameters()) + list(prop_enc.parameters()), lr=5e-4)
    scaler = torch.amp.GradScaler("cuda")

    def z_of(ep, t):  # DINO-WM token = [visual | proprio emb | action emb], copied onto all 196 patches
        v = visual(variant, ep, t)
        p = prop_enc(torch.zeros(len(ep), 1, 1, device=dev)).expand(-1, 196, -1)                # dummy proprio
        a = act_enc(acts[ep, t.clamp(max=T_ACT - 1)].unsqueeze(1)).expand(-1, 196, -1)
        return torch.cat([v, p, a], -1)

    def losses(ep, t):
        z_in, z_tgt = z_of(ep, t), z_of(ep, t + 1).detach()
        z_hat = pred(z_in)
        out = {"loss": F.mse_loss(z_hat[..., :-10], z_tgt[..., :-10]),        # DINO-WM: all but the action dims
               "visual": F.mse_loss(z_hat[..., :D], z_tgt[..., :D]),
               "copy_baseline": F.mse_loss(z_in[..., :D], z_tgt[..., :D])}  # guess "next frame = this frame"
        if variant == "concat":
            out["rgb_half"] = F.mse_loss(z_hat[..., :384], z_tgt[..., :384])
            out["depth_half"] = F.mse_loss(z_hat[..., 384:768], z_tgt[..., 384:768])
        return out

    log = {"variant": variant, "D": D, "params_M": sum(p.numel() for p in pred.parameters()) / 1e6, "train": [], "val": []}
    step, t0 = 0, time.time()
    for epoch in range(epochs):
        pred.train()
        for b in tr_pairs[torch.randperm(len(tr_pairs))].split(bs):
            ep, t = b[:, 0].to(dev), b[:, 1].to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                L = losses(ep, t)
            opt_p.zero_grad(); opt_a.zero_grad()
            scaler.scale(L["loss"]).backward(); scaler.step(opt_p); scaler.step(opt_a); scaler.update()
            step += 1
            if step % 10 == 0: log["train"].append({"step": step, **{k: v.item() for k, v in L.items()}})
        pred.eval(); agg = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for b in va_pairs.split(64):
                for k, v in losses(b[:, 0].to(dev), b[:, 1].to(dev)).items():
                    agg[k] = agg.get(k, 0.0) + v.item() * len(b)
        val = {k: v / len(va_pairs) for k, v in agg.items()}
        val["relative"] = val["visual"] / val["copy_baseline"]
        log["val"].append({"epoch": epoch + 1, "step": step, **val})
        extra = f" | rgb_half {val['rgb_half']:.4f} depth_half {val['depth_half']:.4f}" if variant == "concat" else ""
        print(f"[{variant}] epoch {epoch + 1} step {step} | val visual {val['visual']:.4f} | copy {val['copy_baseline']:.4f} | relative {val['relative']:.3f}{extra} | {time.time() - t0:.0f}s", flush=True)
    log["time_s"] = time.time() - t0
    return log

print(f"train pairs {len(tr_pairs)}, val pairs {len(va_pairs)}")
logs = {v: run(v, args.epochs, args.batch) for v in args.variants.split(",")}

# 3) Results: curves + summary table, saved to --out
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for v, lg in logs.items():
    ax[0].plot([r["step"] for r in lg["train"]], [r["visual"] for r in lg["train"]], label=f"{v} (D={lg['D']})")
    ax[1].plot([r["epoch"] for r in lg["val"]], [r["relative"] for r in lg["val"]], marker="o", label=v)
ax[0].set(title="Train loss on visual tokens (scales differ across variants)", xlabel="step", ylabel="MSE", yscale="log")
ax[1].set(title="Val loss / 'next = current' loss (comparable; lower is better)", xlabel="epoch", ylabel="relative loss")
ax[1].axhline(1.0, ls="--", c="gray"); [a.legend() for a in ax]; plt.tight_layout()
plt.savefig(f"{args.out}/curves.png", dpi=120)
print(f"{'variant':8} {'token D':>7} {'pred M':>7} {'val MSE':>9} {'copy MSE':>9} {'relative':>9} {'time s':>7}")
for v, lg in logs.items():
    f = lg["val"][-1]
    print(f"{v:8} {lg['D']:7d} {lg['params_M']:7.1f} {f['visual']:9.4f} {f['copy_baseline']:9.4f} {f['relative']:9.3f} {lg['time_s']:7.0f}")
if "concat" in logs and "rgb" in logs:
    c, r = logs["concat"]["val"][-1], logs["rgb"]["val"][-1]
    print(f"\nSame-space check (both in DINOv2 space): RGB-only val {r['visual']:.4f} vs RGB half of concat {c['rgb_half']:.4f}")
json.dump({"args": vars(args), "N_EP": N_EP, "s_rgb": s_rgb, "s_dep": s_dep, "scale_dep": scale_dep, "logs": logs},
          open(f"{args.out}/logs.json", "w"), indent=1)
print("saved to", args.out)
