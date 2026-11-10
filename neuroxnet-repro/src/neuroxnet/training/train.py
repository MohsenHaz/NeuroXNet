"""Training loop: 8 studies per step, each study forwarded as 128 slices, slice logits averaged to a study logit
before the loss (Section 3.13). AdamW + cosine annealing (Eq. 20), early stopping on validation loss (patience 15),
checkpoint selection on validation AUROC (detection) and macro-F1 (severity). Writes a prediction CSV for every split."""
import math, json, time, subprocess, pathlib, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, get_worker_info
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import roc_auc_score, f1_score
from ..data.dataset import CTStudyDataset, class_counts
from ..models import build
from ..inference.aggregate import aggregate_logits
from .losses import total_loss, class_weights_inverse_frequency


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    import random
    random.seed(worker_seed); np.random.seed(worker_seed)
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "gen"):
        info.dataset.gen.manual_seed(worker_seed)

def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"

def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def forward_study(model, x, chunk):
    """x: [S,H,W] -> study-level (det_logit, sev_logits, fagg_mean). Slices forwarded in chunks to bound memory."""
    dets, sevs, faggs = [], [], []
    for i in range(0, x.shape[0], chunk):
        xin = x[i:i + chunk, None]
        if model.training and torch.is_grad_enabled() and xin.requires_grad is False:
            # Checkpoint model activations to reduce peak memory on 40-GB GPUs.
            def _fwd(z):
                dd, ss, ff = model(z)
                if ss is None: ss = torch.empty((0,), device=dd.device, dtype=dd.dtype)
                return dd, ss, ff
            d, s, f = checkpoint(_fwd, xin, use_reentrant=False)
            if s.numel() == 0: s = None
        else:
            d, s, f = model(xin)
        dets.append(d); faggs.append(f.mean(0, keepdim=True))
        if s is not None: sevs.append(s)
    det = torch.cat(dets); sev = torch.cat(sevs) if sevs else None
    dl, sl = aggregate_logits(det, sev)
    return dl, sl, torch.cat(faggs).mean(0, keepdim=True)

@torch.no_grad()
def predict_split(model, cfg, ids, split, seed, device, out_csv=None):
    model.eval()
    ds = CTStudyDataset(cfg, ids, train=False)
    rows = []
    for x, yb, ys, sid in DataLoader(ds, batch_size=1, num_workers=2):
        x = x[0].to(device)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            dl, sl, _ = forward_study(model, x, cfg["train"]["slice_chunk"])
        ph = float(torch.sigmoid(dl.float()))
        ps = torch.softmax(sl.float(), 0).cpu().numpy() if sl is not None else np.full(4, np.nan)
        rows.append(dict(study_id=sid[0], split=split, y_bin=int(yb), y_sev=int(ys), p0=ps[0], p1=ps[1], p2=ps[2], p3=ps[3],
                         p_hydro=ph, pred_bin=int(ph >= cfg["eval"]["threshold"]), pred_sev=(int(np.argmax(ps)) if not np.isnan(ps[0]) else -1),
                         seed=seed, git_hash=git_hash()))
    df = pd.DataFrame(rows)
    if out_csv:
        pathlib.Path(out_csv).parent.mkdir(parents=True, exist_ok=True); df.to_csv(out_csv, index=False)
    return df

def validation_metrics(df):
    out = {"val_auroc": roc_auc_score(df.y_bin, df.p_hydro) if df.y_bin.nunique() > 1 else float("nan")}
    if df.pred_sev.min() >= 0:
        out["val_macro_f1"] = f1_score(df.y_sev, df.pred_sev, average="macro")
    return out

def train(cfg, split_ids, out_dir, pred_prefix, seed=42, device=None, log=print):
    """split_ids: dict(train=[...], val=[...], test=[...]). Returns the path of the best checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = build(cfg).to(device)
    n_pos, n_neg, sev_counts = class_counts(cfg, split_ids["train"])
    w_p, w_n = class_weights_inverse_frequency(n_pos, n_neg)
    T = cfg["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr_max"], weight_decay=T["weight_decay"])
    train_ds = CTStudyDataset(cfg, split_ids["train"], train=True, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=T["batch_studies"], shuffle=True, num_workers=T.get("num_workers", 2), drop_last=False,
                        generator=generator, worker_init_fn=_seed_worker, persistent_workers=T.get("num_workers", 2) > 0)
    accum = max(1, int(T.get("grad_accumulation_steps", 1)))   # effective batch = batch_studies x accum (Table 4: 8 studies per step)
    total_steps = T["epochs"] * math.ceil(len(loader) / accum)  # the scheduler advances once per optimiser step
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (T["lr_min"] + 0.5 * (T["lr_max"] - T["lr_min"]) * (1 + math.cos(math.pi * s / total_steps))) / T["lr_max"])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_val, best_sel, bad, history = float("inf"), None, 0, []
    ckpt = out_dir / f"{pred_prefix}.pt"
    for epoch in range(T["epochs"]):
        model.train(); t0 = time.time(); run = 0.0
        opt.zero_grad(set_to_none=True)
        for it, (x, yb, ys, _) in enumerate(loader):
            x, yb, ys = x.to(device), yb.to(device), ys.to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                dls, sls, fs = zip(*[forward_study(model, x[b], T["slice_chunk"]) for b in range(x.shape[0])])
                dl = torch.stack(dls); sl = torch.stack(sls) if sls[0] is not None else None; fagg = torch.cat(fs)
                loss, parts = total_loss(cfg, model, dl.float(), None if sl is None else sl.float(), fagg.float(), yb, ys, w_p, w_n, sev_counts)
            scaler.scale(loss / accum).backward()
            if (it + 1) % accum == 0 or (it + 1) == len(loader):
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
            run += loss.item()
        # validation loss and selection metrics
        vdf = predict_split(model, cfg, split_ids["val"], "val", seed, device)
        vm = validation_metrics(vdf)
        vl = float(-np.mean(np.log(np.clip(np.where(vdf.y_bin == 1, vdf.p_hydro, 1 - vdf.p_hydro), 1e-6, 1))))
        if vdf.pred_sev.min() >= 0:
            vl += 1.5 * float(-np.mean(np.log(np.clip(vdf[["p0", "p1", "p2", "p3"]].values[np.arange(len(vdf)), vdf.y_sev], 1e-6, 1))))
        sel = vm.get("val_auroc", 0) + vm.get("val_macro_f1", 0)
        history.append(dict(epoch=epoch + 1, train_loss=run / len(loader), val_loss=vl, **vm, lr=sched.get_last_lr()[0], seconds=time.time() - t0))
        log(json.dumps(history[-1]))
        if best_sel is None or sel > best_sel:
            best_sel = sel; torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch + 1, "seed": seed, "git": git_hash()}, ckpt)
        if vl < best_val - 1e-6:
            best_val, bad = vl, 0
        else:
            bad += 1
            if bad >= T["patience"]:
                log(f"early stopping at epoch {epoch + 1}"); break
    pd.DataFrame(history).to_csv(out_dir / f"{pred_prefix}_history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    for split in ("val", "test"):
        if split_ids.get(split):
            predict_split(model, cfg, split_ids[split], split, seed, device, out_csv=pathlib.Path("predictions") / split_ids.get("_pred_dir", "internal_test") / f"{pred_prefix}{'' if split == 'test' else '_val'}.csv")
    return ckpt
