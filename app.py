import streamlit as st
import pandas as pd
from pricing_engine import (
    compute_base_price,
    compute_scores,
    compute_pricing_power,
    compute_lambda,
    compute_scenarios,
    optimize_price_ladder,
    extract_volume,
    psychological_round,
)

DATA_PATH = "data/products.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")
    # Derive size columns from product name (e.g. "40 MG 100" → dose=40, unit=MG, qty=100, volume=4000)
    parsed = df["product_name"].apply(extract_volume)
    df["dose"]   = parsed.apply(lambda x: x[0])
    df["unit"]   = parsed.apply(lambda x: x[1])
    df["qty"]    = parsed.apply(lambda x: x[2])
    df["volume"] = parsed.apply(lambda x: x[3])   # total content in base units
    return df


def _scenario_prices_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute scenario prices for every row in df. Returns df with added columns."""
    rows = []
    for _, r in df.iterrows():
        base   = compute_base_price(r)
        scores = compute_scores(r)
        scen   = compute_scenarios(r, scores, base["p_base"])
        rows.append({
            "product_name":   r["product_name"],
            "volume":         r["volume"],
            "price_current":  r["price_current"],
            "dose":           r["dose"],
            "unit":           r["unit"],
            "qty":            r["qty"],
            "p_conservative": scen["conservative"],
            "p_base":         scen["base"],
            "p_aggressive":   scen["aggressive"],
        })
    return pd.DataFrame(rows)


def main():
    st.title("PriceMe – Pricing Recommendation Engine")

    df = load_data()

    # ── Product selection ────────────────────────────────────────────────────
    product_name = st.selectbox("Select a product", df["product_name"].sort_values())
    row = df[df["product_name"] == product_name].iloc[0]

    # ── Product information ──────────────────────────────────────────────────
    st.subheader("Product Information")
    info = row[[
        "product_id", "category", "brand",
        "price_current", "cogs", "inflation", "beta",
        "market_share_12m", "delta_market_share",
        "category_growth_12m", "price_market",
        "rotation_index", "revenue_index",
        "dose", "unit", "qty", "volume",
    ]].rename({
        "product_id": "ID", "category": "Category", "brand": "Brand",
        "price_current": "Current Price (€)", "cogs": "COGS (€)",
        "inflation": "Inflation", "beta": "Beta",
        "market_share_12m": "Market Share (12m)", "delta_market_share": "Δ Market Share",
        "category_growth_12m": "Category Growth (12m)", "price_market": "Market Price (€)",
        "rotation_index": "Rotation Index", "revenue_index": "Revenue Index",
        "dose": "Dose", "unit": "Unit", "qty": "Units/Pack", "volume": "Total Volume",
    })
    st.table(info)

    # ── Layer 1: Base price ──────────────────────────────────────────────────
    base   = compute_base_price(row)
    scores = compute_scores(row)

    st.subheader("Layer 1 — Base Price")
    c1, c2, c3 = st.columns(3)
    c1.metric("Elasticity",     base["elasticity_bucket"])
    c2.metric("Margin Target",  f"{base['margin_target']:.1%}")
    c3.metric("Base Price (€)", f"{base['p_base']:.2f}")

    # ── Layer 2: Scores ──────────────────────────────────────────────────────
    st.subheader("Layer 2 — Product Scores (0–100)")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Brand",           scores["S_brand"],           help="0.5·MS + 0.3·ΔMS + 0.2·SOV  (SOV ≈ market share proxy)")
    s2.metric("Growth",          scores["S_growth"],          help="clip(500·growth_12m + 50, 0, 100)")
    s3.metric("Competitiveness", scores["S_competitiveness"], help="Price index vs. market: <0.95→80, parity→50, >1.05→20")
    s4.metric("Health",          scores["S_health"],          help="50 base − 20 per weak index (rotation, revenue)")

    # ── Layer 3: Pricing power & λ ───────────────────────────────────────────
    s_power = compute_pricing_power(scores)
    lam     = compute_lambda(s_power)

    st.subheader("Layer 3 — Pricing Power & λ")
    p1, p2 = st.columns(2)
    p1.metric("S_power", s_power, help="Weighted average of the 4 scores (centre = 50)")
    p2.metric("λ (lambda)", f"{lam:.4f}", help="σ(k·(S_power−50)) — how much of the gap to P_base we close")

    # ── Layer 4: Scenario prices ─────────────────────────────────────────────
    scenarios = compute_scenarios(row, scores, base["p_base"])
    p0        = float(row["price_current"])

    st.subheader("Layer 4 — Price Scenarios (with guardrails)")
    e1, e2, e3 = st.columns(3)
    for col, (name, price) in zip([e1, e2, e3], scenarios.items()):
        delta_pct = (price - p0) / p0 * 100
        col.metric(name.capitalize(), f"€ {price:.2f}", f"{delta_pct:+.1f}%")

    # ── Layer 5: Brand portfolio price ladder ────────────────────────────────
    st.subheader("Layer 5 — Brand Portfolio & Price Ladder")

    brand      = row["brand"]
    brand_df   = df[df["brand"] == brand].copy()
    portfolio  = _scenario_prices_for_df(brand_df)

    scenario_choice = st.selectbox(
        "Scenario", ["conservative", "base", "aggressive"], index=1
    )

    records = portfolio.to_dict("records")
    ladder  = optimize_price_ladder(records, scenario=scenario_choice)

    # Build display table
    display = portfolio[["product_name", "dose", "unit", "qty", "volume",
                          "price_current", f"p_{scenario_choice}"]].copy()
    display["ladder_price"] = display["product_name"].map(ladder)
    display["price_per_unit"] = display.apply(
        lambda r: round(r["ladder_price"] / r["volume"], 4) if pd.notna(r["volume"]) and r["volume"] else None,
        axis=1,
    )
    display = display.rename(columns={
        "product_name":      "Product",
        "dose":              "Dose",
        "unit":              "Unit",
        "qty":               "Pack Size",
        "volume":            "Total Volume",
        "price_current":     "Current (€)",
        f"p_{scenario_choice}": "Scenario Pre-ladder (€)",
        "ladder_price":      "Ladder Price (€)",
        "price_per_unit":    "Price / Unit",
    })
    st.dataframe(
        display.sort_values("Total Volume"),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
