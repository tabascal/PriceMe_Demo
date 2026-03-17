import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pricing_engine import (
    compute_base_price, compute_scores,
    compute_pricing_power, compute_lambda,
    compute_scenarios, optimize_price_ladder,
    extract_volume,
)

st.set_page_config(page_title="PriceMe", layout="wide")

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "primary":      "#1B3A6B",
    "conservative": "#3498DB",
    "base":         "#27AE60",
    "aggressive":   "#E67E22",
    "low":          "#E74C3C",
    "medium":       "#F39C12",
    "high":         "#27AE60",
    "muted":        "#7F8C8D",
    "bg":           "#F4F6FA",
}

st.markdown(f"""
<style>
    .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
    h1 {{ color: {C['primary']}; }}
    h3 {{ color: {C['primary']}; border-bottom: 2px solid {C['primary']}22;
          padding-bottom: 6px; margin-top: 0.5rem; }}
    .label {{
        font-size: 0.72rem; font-weight: 700; letter-spacing: 1.2px;
        text-transform: uppercase; color: {C['muted']}; margin-bottom: 4px;
    }}
    .badge {{
        display: inline-block; padding: 5px 18px; border-radius: 20px;
        font-weight: 700; font-size: 1rem; letter-spacing: 0.5px;
    }}
    .badge-Low    {{ background:#FADBD8; color:#C0392B; }}
    .badge-Medium {{ background:#FDEBD0; color:#D35400; }}
    .badge-High   {{ background:#D5F5E3; color:#1E8449; }}
    .kpi-card {{
        background: white; border-radius: 10px; padding: 14px 18px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    }}
    .kpi-val  {{ font-size: 1.6rem; font-weight: 700; color: {C['primary']}; margin: 2px 0; }}
    .sc-card  {{
        border-radius: 10px; padding: 20px 16px; text-align: center;
        background: white; box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    }}
    .sc-label {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 1.2px;
                 text-transform: uppercase; margin-bottom: 6px; }}
    .sc-price {{ font-size: 2.1rem; font-weight: 800; color: {C['primary']}; margin: 4px 0; }}
    .sc-delta {{ font-size: 1rem; font-weight: 600; }}
    .arrow    {{ font-size: 1.3rem; margin: 0 8px; }}
</style>
""", unsafe_allow_html=True)

DATA_PATH = "data/products.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=";")
    parsed       = df["product_name"].apply(extract_volume)
    df["dose"]   = parsed.apply(lambda x: x[0])
    df["unit"]   = parsed.apply(lambda x: x[1])
    df["qty"]    = parsed.apply(lambda x: x[2])
    df["volume"] = parsed.apply(lambda x: x[3])
    return df


def _portfolio_scenarios(df):
    rows = []
    for _, r in df.iterrows():
        base  = compute_base_price(r)
        sc    = compute_scores(r)
        scen  = compute_scenarios(r, sc, base["p_base"])
        rows.append({
            "product_name":   r["product_name"],
            "volume":         r["volume"],
            "price_current":  r["price_current"],
            "dose":           r["dose"], "unit": r["unit"], "qty": r["qty"],
            "p_conservative": scen["conservative"],
            "p_base":         scen["base"],
            "p_aggressive":   scen["aggressive"],
        })
    return pd.DataFrame(rows)


# ── Charts ────────────────────────────────────────────────────────────────────

def radar_chart(scores):
    cats = ["Brand", "Growth", "Competitiveness", "Health"]
    vals = [scores["S_brand"], scores["S_growth"],
            scores["S_competitiveness"], scores["S_health"]]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself",
        fillcolor=f"rgba(27,58,107,0.12)",
        line=dict(color=C["primary"], width=2.5),
        marker=dict(size=6, color=C["primary"]),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=C["bg"],
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont_size=9, gridcolor="#DDD"),
            angularaxis=dict(tickfont=dict(size=13, color=C["primary"])),
        ),
        showlegend=False, height=310,
        margin=dict(l=55, r=55, t=35, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def gauge_chart(s_power):
    color = C["low"] if s_power < 40 else (C["medium"] if s_power < 60 else C["high"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=s_power,
        number={"font": {"size": 40, "color": color, "family": "sans-serif"},
                "suffix": " / 100"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CCC",
                     "tickfont": {"size": 10}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "white", "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#FADBD8"},
                {"range": [40, 60], "color": "#FDEBD0"},
                {"range": [60,100], "color": "#D5F5E3"},
            ],
        },
    ))
    fig.update_layout(
        height=230, margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def scenario_bar(p0, scenarios):
    labels  = ["Current", "Conservative", "Base", "Aggressive"]
    prices  = [p0, scenarios["conservative"], scenarios["base"], scenarios["aggressive"]]
    colors  = [C["muted"], C["conservative"], C["base"], C["aggressive"]]
    fig = go.Figure(go.Bar(
        x=labels, y=prices,
        marker_color=colors,
        marker_line_width=0,
        text=[f"€ {p:.2f}" for p in prices],
        textposition="outside",
        textfont=dict(size=13, color=C["primary"], family="sans-serif"),
        width=0.45,
    ))
    fig.update_layout(
        height=310,
        showlegend=False,
        yaxis=dict(range=[min(prices) * 0.92, max(prices) * 1.09],
                   showgrid=True, gridcolor="#EEE",
                   title="Price (€)", titlefont_color=C["muted"]),
        xaxis=dict(tickfont=dict(size=13)),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=20, t=30, b=10),
    )
    return fig


def score_bars_html(scores):
    items = [
        ("Brand",           "S_brand",           C["primary"]),
        ("Growth",          "S_growth",          C["base"]),
        ("Competitiveness", "S_competitiveness", C["conservative"]),
        ("Health",          "S_health",          C["aggressive"]),
    ]
    html = ""
    for label, key, color in items:
        v = scores[key]
        html += f"""
        <div style="margin-bottom:14px">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-size:0.82rem;font-weight:600;color:#444">{label}</span>
                <span style="font-size:0.9rem;font-weight:800;color:{color}">{v}</span>
            </div>
            <div style="background:#E8ECF0;border-radius:6px;height:9px">
                <div style="background:{color};width:{v}%;height:9px;border-radius:6px;
                            transition:width 0.4s ease"></div>
            </div>
        </div>"""
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("PriceMe — Pricing Recommendation Engine")

    df = load_data()
    product_name = st.selectbox("Select a product", df["product_name"].sort_values())
    row = df[df["product_name"] == product_name].iloc[0]
    p0  = float(row["price_current"])

    st.divider()

    # ── Product snapshot ─────────────────────────────────────────────────────
    st.subheader("Product Overview")
    cols = st.columns(6)
    snapshot = [
        ("Brand",        row["brand"]),
        ("Category",     row["category"]),
        ("Current Price",f"€ {p0:.2f}"),
        ("Market Price", f"€ {row['price_market']:.2f}"),
        ("Beta (β)",     row["beta"]),
        ("Market Share", f"{row['market_share_12m']:.0%}"),
    ]
    for col, (label, val) in zip(cols, snapshot):
        col.markdown(
            f'<div class="kpi-card"><div class="label">{label}</div>'
            f'<div class="kpi-val">{val}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Layer 1 ───────────────────────────────────────────────────────────────
    st.subheader("Layer 1 — Anchor Price")
    base   = compute_base_price(row)
    bucket = base["elasticity_bucket"]
    diff   = base["p_base"] - p0
    arrow  = "↑" if diff >= 0 else "↓"
    d_color = C["high"] if diff >= 0 else C["low"]

    b1, b2, b3, b4 = st.columns([1.2, 1, 1, 1.8])
    with b1:
        st.markdown('<div class="label">Elasticity</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="badge badge-{bucket}">{bucket}</span>'
                    f'<div style="color:{C["muted"]};font-size:0.8rem;margin-top:4px">β = {row["beta"]}</div>',
                    unsafe_allow_html=True)
    b2.metric("Margin Target", f"{base['margin_target']:.1%}")
    b3.metric("Base Price",    f"€ {base['p_base']:.2f}")
    with b4:
        st.markdown(
            f'<div style="padding:10px 0;line-height:2">'
            f'<span style="color:{C["muted"]};font-size:0.85rem">Current&nbsp;</span>'
            f'<span style="font-size:1.15rem;font-weight:700">€ {p0:.2f}</span>'
            f'<span class="arrow" style="color:{d_color}">{arrow}</span>'
            f'<span style="font-size:1.15rem;font-weight:700">€ {base["p_base"]:.2f}</span>'
            f'&nbsp;<span style="color:{d_color};font-weight:700">({diff:+.2f} €)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Layer 2 ───────────────────────────────────────────────────────────────
    st.subheader("Layer 2 — Product Scores")
    scores = compute_scores(row)
    col_radar, col_bars = st.columns([1.4, 1])
    with col_radar:
        st.plotly_chart(radar_chart(scores), use_container_width=True)
    with col_bars:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(score_bars_html(scores), unsafe_allow_html=True)

    st.divider()

    # ── Layer 3 ───────────────────────────────────────────────────────────────
    st.subheader("Layer 3 — Pricing Power")
    s_power = compute_pricing_power(scores)
    lam     = compute_lambda(s_power)
    lam_color = C["low"] if lam < 0.4 else (C["medium"] if lam < 0.6 else C["high"])

    col_gauge, col_lam = st.columns([1.3, 1])
    with col_gauge:
        st.plotly_chart(gauge_chart(s_power), use_container_width=True)
    with col_lam:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="label">Lambda (λ)</div>'
            f'<div style="font-size:2.8rem;font-weight:800;color:{lam_color};line-height:1.1">{lam:.3f}</div>'
            f'<div style="color:{C["muted"]};font-size:0.88rem;margin:6px 0 10px">'
            f'Closes <strong>{lam*100:.0f}%</strong> of the gap between<br>current price and base price</div>'
            f'<div style="background:#E8ECF0;border-radius:6px;height:10px">'
            f'<div style="background:{lam_color};width:{lam*100:.0f}%;height:10px;'
            f'border-radius:6px"></div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Layer 4 ───────────────────────────────────────────────────────────────
    st.subheader("Layer 4 — Price Scenarios")
    scenarios = compute_scenarios(row, scores, base["p_base"])

    sc1, sc2, sc3 = st.columns(3)
    for col, (label, key, color) in zip(
        [sc1, sc2, sc3],
        [("Conservative","conservative",C["conservative"]),
         ("Base",        "base",        C["base"]),
         ("Aggressive",  "aggressive",  C["aggressive"])],
    ):
        price     = scenarios[key]
        delta_pct = (price - p0) / p0 * 100
        delta_abs = price - p0
        arr       = "↑" if delta_pct >= 0 else "↓"
        dc        = C["high"] if delta_pct >= 0 else C["low"]
        col.markdown(
            f'<div class="sc-card" style="border-top:4px solid {color}">'
            f'<div class="sc-label" style="color:{color}">{label}</div>'
            f'<div class="sc-price">€ {price:.2f}</div>'
            f'<div class="sc-delta" style="color:{dc}">'
            f'{arr} {delta_pct:+.1f}%&ensp;({delta_abs:+.2f} €)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(scenario_bar(p0, scenarios), use_container_width=True)

    st.divider()

    # ── Layer 5 ───────────────────────────────────────────────────────────────
    st.subheader("Layer 5 — Brand Portfolio & Price Ladder")
    brand_df  = df[df["brand"] == row["brand"]].copy()
    portfolio = _portfolio_scenarios(brand_df)

    scenario_choice = st.selectbox(
        "Scenario", ["conservative", "base", "aggressive"], index=1
    )
    ladder = optimize_price_ladder(portfolio.to_dict("records"), scenario=scenario_choice)

    display = portfolio[["product_name", "dose", "unit", "qty", "volume",
                          "price_current", f"p_{scenario_choice}"]].copy()
    display["Ladder Price"]  = display["product_name"].map(ladder)
    display["Δ vs Current"]  = (display["Ladder Price"] - display["price_current"]).round(2)
    display["€ / Unit"]      = display.apply(
        lambda r: round(r["Ladder Price"] / r["volume"], 4)
        if pd.notna(r["volume"]) and r["volume"] else None, axis=1,
    )
    display = display.rename(columns={
        "product_name":         "Product",
        "dose": "Dose", "unit": "Unit", "qty": "Pack",
        "volume":               "Volume",
        "price_current":        "Current (€)",
        f"p_{scenario_choice}": "Pre-ladder (€)",
    })

    def _color_delta(val):
        if isinstance(val, (int, float)):
            if val > 0:  return "color: #27AE60; font-weight: 600"
            if val < 0:  return "color: #E74C3C; font-weight: 600"
        return ""

    styled = (
        display.sort_values("Volume")
        .style.map(_color_delta, subset=["Δ vs Current"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
