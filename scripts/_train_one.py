"""Train one configuration on the fixed 70/15/15 split and write predictions/<dir>/<name>.csv"""
import argparse, sys, pathlib
sys.path.insert(0, "src")
from neuroxnet.config import load_config
from neuroxnet.data.dataset import read_split
from neuroxnet.training.train import train
ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--pred-dir", default="internal_test"); ap.add_argument("--ckpt-dir", default="checkpoints"); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args(); cfg = load_config(a.config); ids = read_split(cfg); ids["_pred_dir"] = a.pred_dir
log = open(f"logs/{a.name}.log", "a")
def both(s): print(s); log.write(s + "\n"); log.flush()
ck = train(cfg, ids, a.ckpt_dir, a.name, seed=a.seed, log=both)
both(f"checkpoint {ck}")
