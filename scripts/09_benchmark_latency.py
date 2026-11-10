"""Table 14: parameters (feature extractor only), FLOPs per 256x256 slice, GPU latency per complete 128-slice study including
windowing, resizing and slice selection, for every model. Same device, same thread settings, 20 warm-up and 100 timed runs."""
import sys, time, json, pathlib, numpy as np, pandas as pd, torch
sys.path.insert(0, "src")
from neuroxnet.config import load_config
from neuroxnet.models import build
from neuroxnet.models.neuroxnet import count_parameters
from neuroxnet.data.preprocess import preprocess_volume
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfgs = {"NeuroXNet": "configs/neuroxnet.yaml", **{n: f"configs/baselines/{b}.yaml" for n, b in
        [("ResNet50", "resnet50"), ("VGG16", "vgg16"), ("DenseNet121", "densenet121"), ("EfficientNetB0", "effb0"), ("EfficientNetB4", "effb4"), ("TransUNet", "transunet"), ("Attention U-Net", "attunet")]}}
def flops_per_slice(model):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            model(torch.zeros(1, 1, 256, 256, device=device))
        return fc.get_total_flops() / 2 / 1e9          # multiply-accumulates in GFLOPs, the convention used in the manuscript
    except Exception:
        return float("nan")
rows = []; fake_hu = np.random.default_rng(0).normal(30, 20, (160, 512, 512)).astype(np.float32)
for name, path in cfgs.items():
    cfg = load_config(path); model = build(cfg).to(device).eval()
    params = count_parameters(model, feature_extractor_only=hasattr(model, "msam"))
    if not hasattr(model, "msam"): params = sum(p.numel() for p in model.backbone.parameters())
    with torch.no_grad():
        def run():
            x, _ = preprocess_volume(fake_hu, cfg); x = torch.from_numpy(x).to(device)[:, None]
            model(x)
            if device.type == "cuda": torch.cuda.synchronize()
        for _ in range(20): run()
        t = []
        for _ in range(100): t0 = time.perf_counter(); run(); t.append(time.perf_counter() - t0)
    rows.append(dict(Method=name, params_M=round(params / 1e6, 1), flops_G_per_slice=round(flops_per_slice(model), 2), gpu_ms_per_study=round(1000 * float(np.median(t)), 1),
                     device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"))
    print(rows[-1])
pathlib.Path("results/tables").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv("results/tables/table14_latency.csv", index=False)
json.dump({"torch": torch.__version__, "cuda": torch.version.cuda, "device": rows[0]["device"], "threads": torch.get_num_threads()}, open("logs/latency_env.json", "w"), indent=2)
