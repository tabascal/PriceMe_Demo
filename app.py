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

# ── Palette — aligned with Theme v2 ──────────────────────────────────────────
C = {
    "primary":      "#1B4D5C",   # deep teal
    "conservative": "#3888D6",   # dataColors[1] blue
    "base":         "#51BD6C",   # dataColors[0] green  /  good
    "aggressive":   "#FDAB89",   # dataColors[16] warm salmon
    "low":          "#CB2D37",   # bad
    "medium":       "#F1CF6A",   # neutral / center
    "high":         "#51BD6C",   # good
    "muted":        "#5F6B6D",   # dataColors[11]
    "bg":           "#F4F6FA",
    # extra theme accents
    "blue_lt":      "#4DACF1",   # dataColors[3]  light blue
    "green_dk":     "#38934E",   # dataColors[2]  dark green
    "green_lt":     "#C4FC9F",   # dataColors[6]  mint
    "blue_bg":      "#B3DCF9",   # dataColors[7]  sky blue
    "teal":         "#4AC5BB",   # dataColors[10] teal
}

st.markdown(f"""
<style>
    * {{ font-family: 'Segoe UI Semibold', 'Segoe UI', helvetica, arial, sans-serif !important; }}
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
        border-left: 4px solid {C['blue_lt']};
    }}
    .kpi-card-green {{ border-left-color: {C['base']} !important; }}
    .kpi-card-teal  {{ border-left-color: {C['teal']} !important; }}
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
        fillcolor="rgba(81,189,108,0.18)",   # theme green, semi-transparent
        line=dict(color=C["green_dk"], width=2.5),
        marker=dict(size=7, color=C["base"], line=dict(color="white", width=1.5)),
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
                {"range": [0,  40], "color": "#FADBD8"},          # low  — red tint
                {"range": [40, 60], "color": "#B3DCF9"},          # mid  — theme sky blue
                {"range": [60,100], "color": "#C4FC9F"},          # high — theme mint green
            ],
        },
    ))
    fig.update_layout(
        height=230, margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def scenario_bar(p0, scenarios):
    labels = ["Conservative", "Base", "Aggressive"]
    prices = [scenarios["conservative"], scenarios["base"], scenarios["aggressive"]]
    colors = [C["conservative"], C["base"], C["aggressive"]]
    fills  = ["rgba(56,136,214,0.15)", "rgba(81,189,108,0.15)", "rgba(253,171,137,0.15)"]
    deltas = [(p - p0) / p0 * 100 for p in prices]

    fig = go.Figure()

    # Background highlight bands per scenario
    for i, (label, fill) in enumerate(zip(labels, fills)):
        fig.add_vrect(x0=i - 0.42, x1=i + 0.42, fillcolor=fill,
                      layer="below", line_width=0)

    # Bars
    fig.add_trace(go.Bar(
        x=labels, y=prices,
        marker=dict(color=colors, line=dict(width=0), opacity=0.92),
        width=0.5,
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>€ %{y:.2f}<extra></extra>",
    ))

    # Price label inside/top of bar
    for label, price, color in zip(labels, prices, colors):
        fig.add_annotation(
            x=label, y=price,
            text=f"<b>€ {price:.2f}</b>",
            showarrow=False, yshift=14,
            font=dict(size=15, color=color),
        )

    # Delta badge below price label
    for label, price, delta in zip(labels, prices, deltas):
        d_color = C["green_dk"] if delta >= 0 else C["low"]
        arrow   = "▲" if delta >= 0 else "▼"
        fig.add_annotation(
            x=label, y=price,
            text=f"<b>{arrow} {delta:+.1f}%</b>",
            showarrow=False, yshift=36,
            font=dict(size=12, color=d_color),
        )

    # Reference line — current price
    fig.add_hline(
        y=p0, line_dash="dash", line_color=C["muted"], line_width=1.8,
        annotation_text=f"<b>Current  € {p0:.2f}</b>",
        annotation_position="right",
        annotation_font=dict(size=11, color=C["muted"]),
    )

    all_vals = prices + [p0]
    fig.update_layout(
        height=380,
        showlegend=False,
        yaxis=dict(
            range=[min(all_vals) * 0.88, max(all_vals) * 1.14],
            showgrid=True, gridcolor="#EEE", gridwidth=1, zeroline=False,
            tickprefix="€ ", tickfont=dict(size=11, color=C["muted"]),
        ),
        xaxis=dict(
            tickfont=dict(size=14, color=C["primary"]),
            showline=False, ticks="",
        ),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=110, t=50, b=20),
        bargap=0.35,
    )
    return fig


def score_bars_html(scores):
    items = [
        ("Brand",           "S_brand",           C["primary"]),
        ("Growth",          "S_growth",          C["green_dk"]),
        ("Competitiveness", "S_competitiveness", C["blue_lt"]),
        ("Health",          "S_health",          C["teal"]),
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
        ("Elasticity (β)", row["beta"]),
        ("Market Share", f"{row['market_share_12m']:.0%}"),
    ]
    accents = ["", "kpi-card-green", "kpi-card-teal", "kpi-card-green", "", "kpi-card-teal"]
    for col, (label, val), extra in zip(cols, snapshot, accents):
        col.markdown(
            f'<div class="kpi-card {extra}"><div class="label">{label}</div>'
            f'<div class="kpi-val">{val}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Layer 1 ───────────────────────────────────────────────────────────────
    st.subheader("Layer 1 — Anchor Price")
    st.caption("We start by calculating the minimum viable price: what the product needs to cost to cover expenses (COGS + inflation) while hitting the margin target. Products with low price sensitivity get a higher target margin.")
    base   = compute_base_price(row)
    bucket = base["elasticity_bucket"]
    diff   = base["p_base"] - p0
    arrow  = "↑" if diff >= 0 else "↓"
    d_color = C["high"] if diff >= 0 else C["low"]

    b1, b2, b3, b4 = st.columns([1.2, 1, 1, 1.8])
    with b1:
        st.markdown('<div class="label">Elasticity</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="badge badge-{bucket}">{bucket}</span>'
                    f'<div style="color:{C["muted"]};font-size:0.8rem;margin-top:4px">Elasticity β = {row["beta"]}</div>',
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
    st.caption("Four dimensions assess the product's market position: brand strength, category growth momentum, price competitiveness vs. the market, and commercial health (rotation & revenue). Each score runs from 0 to 100.")
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
    st.caption("The four scores are combined into a single Pricing Power index (0–100). This drives λ (lambda): how boldly we move from the current price toward the anchor price. A strong product (high score) gets a λ close to 1 — meaning a full move. A weak product stays closer to its current price.")
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
    st.caption("Three pricing strategies are offered — Conservative, Base, and Aggressive — each scaling λ differently. Guardrails are applied automatically: maximum price change caps per scenario, and an additional cap if the product is already above market price and commercially weak.")
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
    st.caption("Prices are reviewed across the full brand portfolio to ensure internal coherence: larger pack sizes must always offer a lower price per unit than smaller ones. An optimisation algorithm adjusts individual prices minimally to enforce this rule, then applies psychological rounding (e.g. €9.95, €12.75).")
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
        lambda r: round(r["Ladder Price"] / r["volume"], 2)
        if pd.notna(r["volume"]) and r["volume"] else None, axis=1,
    )
    display = display.rename(columns={
        "product_name":         "Product",
        "dose": "Dose", "unit": "Unit", "qty": "Pack",
        "volume":               "Volume",
        "price_current":        "Current (€)",
        f"p_{scenario_choice}": "Pre-ladder (€)",
    })

    fmt = {
        "Dose":           "{:.0f}",
        "Volume":         "{:.0f}",
        "Current (€)":    "{:.2f}",
        "Pre-ladder (€)": "{:.2f}",
        "Ladder Price":   "{:.2f}",
        "Δ vs Current":   "{:.2f}",
        "€ / Unit":       "{:.2f}",
    }

    def _color_delta(val):
        if isinstance(val, (int, float)):
            if val > 0:  return "color: #27AE60; font-weight: 600"
            if val < 0:  return "color: #E74C3C; font-weight: 600"
        return ""

    styled = (
        display.sort_values("Volume")
        .style.format(fmt)
        .map(_color_delta, subset=["Δ vs Current"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
