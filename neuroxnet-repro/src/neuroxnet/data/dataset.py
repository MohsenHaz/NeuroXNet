import numpy as np, pandas as pd, torch, pathlib
from torch.utils.data import Dataset
from .augment import augment

class CTStudyDataset(Dataset):
    """One item = one preprocessed study [128,256,256] plus binary and severity labels."""
    def __init__(self, cfg, split_ids, train=False, seed=42):
        self.cfg, self.ids, self.train = cfg, list(split_ids), train
        lab = pd.read_csv(cfg["data"]["label_file"]).set_index("study_id")
        self.y_sev = {i: int(lab.loc[i, "final_label"]) for i in self.ids}
        self.root = pathlib.Path(cfg["data"]["root"])
        self.gen = torch.Generator().manual_seed(seed)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, k):
        sid = self.ids[k]
        x = torch.from_numpy(np.load(self.root / f"{sid}.npy").astype(np.float32))
        if self.train:
            x = augment(x, self.cfg, self.gen)
        ys = self.y_sev[sid]
        return x, torch.tensor(float(ys > 0)), torch.tensor(ys), sid

def read_split(cfg):
    df = pd.read_csv(cfg["data"]["split_file"])
    return {s: df.loc[df.split == s, "study_id"].tolist() for s in ("train", "val", "test")}

def class_counts(cfg, ids):
    lab = pd.read_csv(cfg["data"]["label_file"]).set_index("study_id")
    y = lab.loc[ids, "final_label"].values
    return int((y > 0).sum()), int((y == 0).sum()), np.bincount(y, minlength=4)
