"""Stochastic training augmentation, Eq. (21)-(23): spatial (rotate, scale, translate, flip) then intensity (gamma, contrast, noise)."""
import math, torch, torch.nn.functional as F

def _affine_grid(x, theta_deg, scale, tx, ty):
    S, H, W = x.shape
    th = math.radians(theta_deg)
    A = torch.tensor([[math.cos(th) / scale, -math.sin(th) / scale, tx],
                      [math.sin(th) / scale,  math.cos(th) / scale, ty]], dtype=x.dtype, device=x.device)
    grid = F.affine_grid(A[None].expand(S, 2, 3), (S, 1, H, W), align_corners=False)
    return F.grid_sample(x[:, None], grid, padding_mode="zeros", align_corners=False)[:, 0]

def augment(x, cfg, gen=None):
    """x: [S,H,W] normalised tensor. One random transform per study, applied identically to all slices."""
    a = cfg["augmentation"]; g = gen
    r = lambda lo, hi: float(torch.empty(1).uniform_(lo, hi, generator=g))
    x = _affine_grid(x, r(-a["rotation_deg"], a["rotation_deg"]), r(*a["scale"]), r(-a["translate"], a["translate"]), r(-a["translate"], a["translate"]))
    if r(0, 1) < a["hflip_p"]:
        x = torch.flip(x, dims=[-1])
    # intensity transforms act on the [0,1] scale, so undo and redo the normalisation
    m, s = cfg["data"]["normalisation"]["mean"], cfg["data"]["normalisation"]["std"]
    u = (x * s + m).clamp(0, 1)
    u = u.pow(r(*a["gamma"]))
    u = ((u - u.mean()) * r(*a["contrast"]) + u.mean()).clamp(0, 1)
    u = u + torch.randn(u.shape, generator=g, dtype=u.dtype) * r(0, a["noise_sigma_max"])
    return (u - m) / s
