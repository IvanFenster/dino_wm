# [ARPL-WM] DINOv2 ViT-S/14 weights without Meta's server (dl.fbaipublicfiles.com had an expired TLS
# certificate from 2026-11-11). Takes the same Meta weights from Hugging Face (facebook/dinov2-small),
# renames the keys back to Meta's format, checks they load exactly into the pinned DINOv2 model and give
# Hugging Face's features, saves them to a cache file, and copies that into torch hub's checkpoint folder,
# where models/dino.py's torch.hub.load(..., pretrained=True) finds them instead of downloading.
# Moved from the reproduction notebook (cell "DINOv2 weights"), paths made configurable.
# Usage: python scripts/fetch_dinov2_weights.py [cache_file]
#   cache_file default: $WM/cache/dinov2_vits14_pretrain.pth. Later runs only copy it (instant).
#   Needs safetensors, huggingface_hub and transformers the first time only.
import os, sys, shutil, torch

DINOV2_COMMIT = "85a24602099d397264d5b30461ad7f3bfd726ca1"  # same pin as models/dino.py
WM = os.environ.get("WM", ".")
dst = sys.argv[1] if len(sys.argv) > 1 else f"{WM}/cache/dinov2_vits14_pretrain.pth"
hub = os.path.join(torch.hub.get_dir(), "checkpoints", "dinov2_vits14_pretrain.pth")
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
os.makedirs(os.path.dirname(hub), exist_ok=True)

if not os.path.exists(dst):
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    hf = load_file(hf_hub_download("facebook/dinov2-small", "model.safetensors"))
    sd = {"cls_token": hf["embeddings.cls_token"], "pos_embed": hf["embeddings.position_embeddings"],
          "mask_token": hf["embeddings.mask_token"],
          "patch_embed.proj.weight": hf["embeddings.patch_embeddings.projection.weight"],
          "patch_embed.proj.bias": hf["embeddings.patch_embeddings.projection.bias"],
          "norm.weight": hf["layernorm.weight"], "norm.bias": hf["layernorm.bias"]}
    depth = 1 + max(int(k.split(".")[2]) for k in hf if k.startswith("encoder.layer."))
    for i in range(depth):
        h, m = f"encoder.layer.{i}.", f"blocks.{i}."
        a = h + "attention.attention."
        for t in ("weight", "bias"):
            sd[m + "attn.qkv." + t] = torch.cat([hf[a + "query." + t], hf[a + "key." + t], hf[a + "value." + t]])
            sd[m + "attn.proj." + t] = hf[h + "attention.output.dense." + t]
            for n in ("norm1", "norm2", "mlp.fc1", "mlp.fc2"):
                sd[m + n + "." + t] = hf[h + n + "." + t]
        sd[m + "ls1.gamma"] = hf[h + "layer_scale1.lambda1"]
        sd[m + "ls2.gamma"] = hf[h + "layer_scale2.lambda1"]
    # check 1: loads into the pinned DINOv2 model with no missing, extra or misshaped weights
    model = torch.hub.load(f"facebookresearch/dinov2:{DINOV2_COMMIT}", "dinov2_vits14",
                           pretrained=False, trust_repo=True, skip_validation=True).eval()  # skip GitHub's commit check (rate-limited)
    model.load_state_dict(sd, strict=True)
    # check 2: same patch features as Hugging Face's model (518x518 = native size, so no resizing differences)
    from transformers import Dinov2Model
    ref = Dinov2Model.from_pretrained("facebook/dinov2-small").eval()
    x = torch.randn(1, 3, 518, 518)
    with torch.no_grad():
        diff = (model.forward_features(x)["x_norm_patchtokens"] - ref(pixel_values=x).last_hidden_state[:, 1:]).abs().max().item()
    print(f"max feature difference vs Hugging Face: {diff:.2e}")
    assert diff < 1e-3, "conversion mismatch"
    torch.save(sd, dst)

shutil.copy(dst, hub)
print("DINOv2 weights ready:", hub, f"{os.path.getsize(hub) / 1e6:.0f} MB")
