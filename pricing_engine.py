import re
import numpy as np
from scipy.optimize import minimize

# ── Constants ────────────────────────────────────────────────────────────────

MARGIN_BY_ELASTICITY = {
    "Low":    0.30,
    "Medium": 0.425,
    "High":   0.575,
}

# S_brand weights: MS_12 (0.5), ΔMS (0.3), SOV proxy (0.2)
W_BRAND = {"ms": 0.5, "delta_ms": 0.3, "sov": 0.2}

# S_power weights (must sum to 1)
W_POWER = {"brand": 0.30, "growth": 0.25, "comp": 0.25, "health": 0.20}

SCENARIOS = {
    "conservative": {"k_esc": 0.6, "clamp": (-0.03,  0.05)},
    "base":         {"k_esc": 1.0, "clamp": (-0.05,  0.10)},
    "aggressive":   {"k_esc": 1.3, "clamp": (-0.10,  0.18)},
}

LAMBDA_K = 0.1          # sigmoid steepness — tune to sharpen/flatten λ response
DELTA_BUFFER = 0.50     # min €-gap between adjacent price tiers (anti-cannibalization)

_UNIT_MULTIPLIERS = {
    "MCG": 0.001, "MG": 1, "G": 1_000, "KG": 1_000_000,
    "ML": 1, "L": 1_000, "UI": 1,
}

_PSY_ENDINGS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99]


# ── Volume extraction ────────────────────────────────────────────────────────

def extract_volume(product_name: str):
    """
    Parses '<dose> <unit> [<qty>]' from a product name.

    Handles two formats:
      - Tablets/capsules: '40 MG 100'  → dose=40, unit=MG, qty=100, volume=4000
      - Liquids/gels:     '600 ML'     → dose=600, unit=ML, qty=1,   volume=600

    Returns (dose, unit, qty, total_volume_in_base_units).
    total_volume = dose_base_unit * qty — used for price-per-unit comparisons.
    Returns (None, None, None, None) if no match.
    """
    pattern = r'(\d+(?:[.,]\d+)?)\s*(MCG|MG|ML|KG|UI|G|L)(?:\s+(\d+))?'
    match = re.search(pattern, product_name.upper())
    if not match:
        return None, None, None, None
    dose = float(match.group(1).replace(",", "."))
    unit = match.group(2)
    qty  = int(match.group(3)) if match.group(3) else 1
    volume = dose * _UNIT_MULTIPLIERS.get(unit, 1) * qty
    return dose, unit, qty, volume


# ── Layer 1 — Base price ─────────────────────────────────────────────────────

def classify_elasticity(beta: float) -> str:
    if beta < -2:
        return "Low"
    elif beta < -1.2:
        return "Medium"
    return "High"


def compute_base_price(row) -> dict:
    """
    Layer 1: Anchor price from cost + margin target.
      c_adj  = cogs * (1 + inflation)
      p_base = c_adj / (1 - margin_target)
    """
    bucket = classify_elasticity(row["beta"])
    margin = MARGIN_BY_ELASTICITY[bucket]
    c_adj  = row["cogs"] * (1 + row["inflation"])
    p_base = c_adj / (1 - margin)
    return {
        "elasticity_bucket": bucket,
        "margin_target":     margin,
        "p_base":            round(p_base, 4),
    }


# ── Layer 2 — Scores (0–100) ─────────────────────────────────────────────────

def score_brand(row) -> float:
    """
    S_brand = 0.5*MS_12 + 0.3*ΔMS + 0.2*SOV   (normalized to 0–100)

    SOV (Share of Voice) is not available in the dataset.
    Approximation: SOV ≈ MS_12 (brand visibility correlates with market share).
    ΔMS is shifted by +0.1 to keep typical small-negative values contributing
    positively — the shift is a calibration assumption, not a distortion.
    """
    sov           = row["market_share_12m"]          # SOV proxy
    delta_shifted = row["delta_market_share"] + 0.1  # centre around 0
    raw = (
        W_BRAND["ms"]       * row["market_share_12m"]
        + W_BRAND["delta_ms"] * delta_shifted
        + W_BRAND["sov"]      * sov
    )
    return round(float(np.clip(raw * 100, 0, 100)), 1)


def score_growth(row) -> float:
    """
    S_growth = clip(a * growth_12m + b, 0, 100)

    Calibration: growth values in dataset are small decimals (~±0.1 range).
    a=500 maps [-0.1, +0.1] → [0, 100], giving full-range discrimination.
    b=50 centres neutral growth (0%) at score 50.
    """
    return round(float(np.clip(500 * row["category_growth_12m"] + 50, 0, 100)), 1)


def score_competitiveness(row) -> int:
    """
    PI = price_current / price_market
    PI < 0.95  → 80  (priced below market — competitive)
    PI 0.95–1.05 → 50  (parity)
    PI > 1.05  → 20  (priced above market)
    """
    pi = row["price_current"] / row["price_market"]
    if pi < 0.95:
        return 80
    elif pi <= 1.05:
        return 50
    return 20


def score_health(row) -> int:
    """
    Start at 50. Penalise each index below 1 by -20.
    Range: 10 (both weak) → 50 (both healthy).
    """
    score = 50
    if row["rotation_index"] < 1:
        score -= 20
    if row["revenue_index"] < 1:
        score -= 20
    return score


def compute_scores(row) -> dict:
    return {
        "S_brand":           score_brand(row),
        "S_growth":          score_growth(row),
        "S_competitiveness": score_competitiveness(row),
        "S_health":          score_health(row),
    }


# ── Layer 3 — Pricing power & λ ─────────────────────────────────────────────

def compute_pricing_power(scores: dict) -> float:
    """
    S_power = weighted average of the four scores.
    Centre at 50: S_power=50 → λ≈0.5 (halfway between P0 and P_base).
    """
    s = (
        W_POWER["brand"]  * scores["S_brand"]
        + W_POWER["growth"] * scores["S_growth"]
        + W_POWER["comp"]   * scores["S_competitiveness"]
        + W_POWER["health"] * scores["S_health"]
    )
    return round(s, 1)


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def compute_lambda(s_power: float, k: float = LAMBDA_K) -> float:
    """
    λ = σ(k * (S_power − 50))
    k controls how sharply λ transitions from 0→1 around S_power=50.
    """
    return round(float(_sigmoid(k * (s_power - 50))), 4)


# ── Layer 4 — Scenarios + guardrails ────────────────────────────────────────

def _apply_guardrails(p_scenario: float, p0: float, clamp: tuple,
                      s_health: float, pi: float) -> float:
    lo, hi = clamp
    # Rule 5.2: overpriced + unhealthy → cap increase to +1%
    if pi > 1.05 and s_health < 40:
        hi = min(hi, 0.01)
    return float(np.clip(p_scenario, p0 * (1 + lo), p0 * (1 + hi)))


def compute_scenarios(row, scores: dict, p_base: float) -> dict:
    """
    Layers 4+5.1+5.2: scenario prices with λ scaling and guardrails.
    """
    p0       = float(row["price_current"])
    s_power  = compute_pricing_power(scores)
    lam      = compute_lambda(s_power)
    pi       = p0 / row["price_market"]
    s_health = scores["S_health"]

    out = {}
    for name, cfg in SCENARIOS.items():
        lam_esc = lam * cfg["k_esc"]
        p_pre   = p0 + lam_esc * (p_base - p0)
        p_final = _apply_guardrails(p_pre, p0, cfg["clamp"], s_health, pi)
        out[name] = round(p_final, 4)
    return out


# ── Layer 5 — Psychological rounding ────────────────────────────────────────

def psychological_round(price: float) -> float:
    """Round to the nearest psychological price ending (e.g. .95, .99, .75)."""
    base    = int(price)
    decimal = price - base
    ending  = min(_PSY_ENDINGS, key=lambda e: abs(decimal - e))
    return round(base + ending, 2)


# ── Layer 5 — Price ladder optimisation (portfolio) ─────────────────────────

def optimize_price_ladder(portfolio: list, scenario: str = "base") -> dict:
    """
    Layer 5.3: enforces price-per-unit decreasing with total volume within brand.

    Solves:   min  Σ wᵢ (Pᵢ − Pᵢ_target)²     (wᵢ = 1 for all i)
    Subject to:   PU[i] > PU[i+1]   where PU = P / volume
                  and buffer guard:  P[i+1] ≥ P[i] + δ_buffer (anti-cannibalization)

    Products without a parsed volume are returned at their scenario price
    (psychological-rounded) without entering the optimisation.

    Returns {product_name: final_price}.
    """
    key = f"p_{scenario}"

    # Split into optimisable (has volume) and pass-through
    has_vol  = [x for x in portfolio if x.get("volume")]
    no_vol   = [x for x in portfolio if not x.get("volume")]

    result = {x["product_name"]: psychological_round(x[key]) for x in no_vol}
    if not has_vol:
        return result

    items   = sorted(has_vol, key=lambda x: x["volume"])
    targets = np.array([x[key]      for x in items], dtype=float)
    volumes = np.array([x["volume"] for x in items], dtype=float)
    names   = [x["product_name"] for x in items]
    n       = len(items)

    if n == 1:
        result[names[0]] = psychological_round(targets[0])
        return result

    def objective(prices):
        return float(np.sum((prices - targets) ** 2))

    constraints = []
    for i in range(n - 1):
        # Price-per-unit constraint: PU[i] > PU[i+1]
        constraints.append({
            "type": "ineq",
            "fun": lambda p, i=i: p[i] / volumes[i] - p[i + 1] / volumes[i + 1],
        })
        # Anti-cannibalization: larger pack must be at least δ_buffer cheaper
        constraints.append({
            "type": "ineq",
            "fun": lambda p, i=i: p[i] - p[i + 1] + DELTA_BUFFER,
        })

    bounds = [(max(0.01, t * 0.85), t * 1.15) for t in targets]
    opt = minimize(objective, targets, method="SLSQP",
                   bounds=bounds, constraints=constraints)
    optimised = opt.x if opt.success else targets

    for name, price in zip(names, optimised):
        result[name] = psychological_round(float(price))
    return result
