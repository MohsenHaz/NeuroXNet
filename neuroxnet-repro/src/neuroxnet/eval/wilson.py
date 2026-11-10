import math

def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for k successes in n trials. Returns (lo, hi) as proportions."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

def fmt_pct_ci(k, n, dec=2, cidec=1):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.{dec}f} [{100 * lo:.{cidec}f}–{100 * hi:.{cidec}f}]"
