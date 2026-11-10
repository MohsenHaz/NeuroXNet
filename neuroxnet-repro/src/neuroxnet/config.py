import copy, yaml, pathlib

def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out

def load_config(path):
    """Load a YAML config, resolving the `extends:` chain (child values override parent values)."""
    cfg = yaml.safe_load(open(path))
    if "extends" in cfg:
        parent = load_config(pathlib.Path(cfg.pop("extends")))
        cfg = _merge(parent, cfg)
    return cfg
