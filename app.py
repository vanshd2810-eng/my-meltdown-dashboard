"""My Meltdown — kava café-bar idea validation dashboard.

Run locally:  streamlit run app.py
Data: data/my_meltdown_survey.csv (synthetic). The sidebar can regenerate synthetic data
or take a real survey export with the same columns.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from simulate import generate
from pipeline import run

st.set_page_config(page_title="My Meltdown · Idea validation", page_icon="🍵", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ brand
TEAL, CORAL, INDIGO, ROSE, AMBER, SLATE = "#0F766E", "#F97316", "#4F46E5", "#E11D48", "#D97706", "#94A3B8"
SEG_COLORS = [TEAL, CORAL, INDIGO, ROSE, AMBER, "#0EA5E9", "#84CC16"]
GOOD, BAD, WARN = "#15803D", "#B91C1C", "#B45309"
DATA_PATH = "data/my_meltdown_survey.csv"

pio.templates["meltdown"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, Source Sans Pro, Arial, sans-serif", size=13, color="#1F2937"),
    colorway=SEG_COLORS, plot_bgcolor="white", paper_bgcolor="white",
    title=dict(font=dict(size=15, color="#111827"), x=0, xanchor="left", y=0.97, yanchor="top"),
    xaxis=dict(gridcolor="#EEF2F5", zerolinecolor="#D5DCE3", linecolor="#D5DCE3"),
    yaxis=dict(gridcolor="#EEF2F5", zerolinecolor="#D5DCE3", linecolor="#D5DCE3"),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title_text=""),
    margin=dict(l=10, r=10, t=72, b=10), hoverlabel=dict(bgcolor="white", font_size=12)))
pio.templates.default = "meltdown"

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg, #0F766E 0%, #115E59 55%, #134E4A 100%); color: #fff; border-radius: 14px;
       padding: 22px 28px; margin-bottom: 18px;}
.hero h1 {color: #fff; font-size: 1.9rem; margin: 0 0 4px 0; padding: 0; font-weight: 700;}
.hero p {color: #D1FAE5; margin: 0; font-size: 1rem;}
.badges {margin-top: 12px;}
.badge {display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); color: #fff;
        border-radius: 999px; padding: 3px 11px; font-size: .78rem; margin-right: 6px;}
.verdict {border-radius: 12px; padding: 14px 18px; margin: 4px 0 16px 0; border: 1px solid #FCD34D; background: #FFFBEB;}
.verdict b.t {color: #92400E; font-size: 1.05rem;}
.kpi {background: #fff; border: 1px solid #E5E9EE; border-radius: 12px; padding: 14px 16px; height: 100%; margin-bottom: 4px;
      box-shadow: 0 1px 2px rgba(16,24,40,.04);}
.kpi .l {color: #64748B; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em;}
.kpi .v {font-size: clamp(1.25rem, 2.1vw, 1.75rem); font-weight: 700; color: #111827; line-height: 1.25; margin-top: 2px; white-space: nowrap;}
.kpi .s {color: #64748B; font-size: .8rem; margin-top: 2px;}
.kpi.good {border-top: 4px solid #15803D;} .kpi.bad {border-top: 4px solid #B91C1C;} .kpi.neutral {border-top: 4px solid #0F766E;}
.takeaway {background: #F0FDFA; border-left: 4px solid #0F766E; border-radius: 8px; padding: 12px 16px; margin: 6px 0 14px 0;}
.takeaway b {color: #115E59;}
.sec {font-size: 1.25rem; font-weight: 700; color: #111827; margin: 18px 0 2px 0;}
.sub {color: #64748B; font-size: .92rem; margin-bottom: 10px;}
.persona {background: #fff; border: 1px solid #E5E9EE; border-radius: 12px; padding: 14px 16px; height: 100%; margin-bottom: 14px;}
.persona h4 {margin: 0; font-size: 1.02rem;} .persona .tier {color: #64748B; font-size: .8rem; margin-bottom: 6px;}
.persona .big {font-size: 1.6rem; font-weight: 700;} .persona .desc {font-size: .85rem; color: #374151; margin: 6px 0;}
.persona table {width: 100%; font-size: .8rem;} .persona td {padding: 1px 0;} .persona td.r {text-align: right; font-weight: 600;}
.pill {display: inline-block; border-radius: 999px; padding: 1px 9px; font-size: .75rem; font-weight: 600;}
.pill.pass {background: #DCFCE7; color: #166534;} .pill.fail {background: #FEE2E2; color: #991B1B;}
.pill.high {background: #FEE2E2; color: #991B1B;} .pill.med {background: #FEF3C7; color: #92400E;} .pill.low {background: #E0F2FE; color: #075985;}
table.sc {width: 100%; border-collapse: collapse; font-size: .9rem;}
table.sc th {text-align: left; color: #64748B; font-weight: 600; border-bottom: 1px solid #E5E9EE; padding: 6px 8px;}
table.sc td {border-bottom: 1px solid #F1F4F7; padding: 7px 8px; vertical-align: top;}
.review {background: #fff; border: 1px solid #E5E9EE; border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;}
.review h4 {margin: 0 0 6px 0; font-size: 1rem;} .review p {margin: 3px 0; font-size: .9rem; color: #374151;}
.foot {color: #94A3B8; font-size: .8rem; margin-top: 28px; border-top: 1px solid #EEF2F5; padding-top: 10px;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def load_default():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return generate(3500, 42)


@st.cache_data(show_spinner=False)
def make_synthetic(n, seed):
    return generate(n, seed)


@st.cache_data(show_spinner="Cleaning the data and running the four models…")
def analyse(raw: pd.DataFrame):
    return run(raw)


def kpi(res, metric):
    return next(d["value"] for d in res["kpis"] if d["metric"] == metric)


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def inr(x):
    return f"₹{x:,.0f}"


def card(label, value, sub="", tone="neutral"):
    return f'<div class="kpi {tone}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def takeaway(text):
    st.markdown(f'<div class="takeaway"><b>Key takeaway.</b> {text}</div>', unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="sec">{title}</div>' + (f'<div class="sub">{sub}</div>' if sub else ""), unsafe_allow_html=True)


def chart(fig, h=380):
    fig.update_layout(height=h, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def persona_names(segs):
    names, left = {}, list(range(len(segs)))
    def take(i, name):
        names[segs[i]["segment"]] = name
        left.remove(i)
    if len(segs) >= 4:
        take(max(left, key=lambda i: segs[i]["weekly_drinkers"]), "Night-out drinkers")
        take(min(left, key=lambda i: segs[i]["median_income"]), "Budget students")
        take(max(left, key=lambda i: segs[i]["safety_concern"]), "Cautious skeptics")
        for i in list(left):
            take(i, "Curious professionals" if len(left) == 1 else f"Segment {segs[i]['segment']}")
    else:
        for i in list(left):
            take(i, f"Segment {segs[i]['segment']}")
    return names


PERSONA_DESC = {
    "Curious professionals": "Salaried, mostly metro, lighter drinkers who enjoy trying new things and worry little about safety.",
    "Night-out drinkers": "Go out and drink often and like the idea of a buzz the most. Fewer people, highest value per head.",
    "Budget students": "Students on pocket money. Open to the idea, but price is what holds them back.",
    "Cautious skeptics": "Worried about safety and not keen on new things. Not a launch target.",
}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🍵 My Meltdown")
    st.caption("Kava café-come-bar · idea validation")
    st.divider()
    st.markdown("**Data source**")
    source = st.radio("Data source", ["Bundled synthetic data", "Generate new synthetic data", "Upload real survey CSV"],
                      label_visibility="collapsed")
    if source == "Generate new synthetic data":
        n = st.slider("Respondents", 2000, 5000, 3500, step=250)
        seed = st.number_input("Random seed", 0, 9999, 42)
        raw = make_synthetic(n, int(seed))
    elif source == "Upload real survey CSV":
        up = st.file_uploader("CSV with the same columns as the bundled file", type="csv")
        raw = pd.read_csv(up) if up is not None else load_default()
        if up is None:
            st.info("No file yet: showing the bundled data.")
    else:
        raw = load_default()
    st.download_button("⬇ Download dataset (CSV)", raw.to_csv(index=False).encode(), "my_meltdown_survey.csv",
                       "text/csv", use_container_width=True)
    st.divider()
    st.caption("**About the data.** Synthetic responses generated from assumed customer behaviour, with noise and "
               "outliers added on purpose. They test the survey and the models end to end; real answers are needed "
               "for a real decision.")

res = analyse(raw)
segs = res["clu_segments"]
NAMES = persona_names(segs)
seg_label = {s["segment"]: f"{s['segment']} · {NAMES[s['segment']]}" for s in segs}
base = res["baselines"]
clf_best = next(m for m in res["clf_models"] if m["best"] == 1)
passed = sum(s["pass"] for s in res["scorecard"])
is_synth = source != "Upload real survey CSV" or up is None if source == "Upload real survey CSV" else True

# ------------------------------------------------------------------ hero
st.markdown(f"""
<div class="hero">
  <h1>My Meltdown — kava café-bar idea validation</h1>
  <p>Would Gen Z and young professionals pay for a relaxing, alcohol-free kava café by day and kava bar by night?</p>
  <div class="badges"><span class="badge">{len(raw):,} survey responses</span><span class="badge">{kpi(res, 'respondents'):,} after cleaning</span>
  <span class="badge">30 questions</span><span class="badge">Classification · Clustering · Association rules · Regression</span>
  <span class="badge">{'Synthetic data' if is_synth else 'Uploaded data'}</span></div>
</div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="verdict"><b class="t">Verdict: promising, but not proven.</b>
{pct(kpi(res, 'buy_rate'))} (±{pct(base['buy_rate_ci'])}) of respondents say they would buy and {passed} of {len(res['scorecard'])}
viability checks pass. Word of mouth is the weak spot (buyer NPS {kpi(res, 'nps_buyers'):.1f}). Stated intent and synthetic data
both overstate certainty — see <i>Critical review</i>.</div>""", unsafe_allow_html=True)

tabs = st.tabs(["📊 Executive summary", "🧹 Data quality", "🎯 Classification", "👥 Segmentation", "🧺 Service bundles",
                "💰 Spend model", "🏷️ Pricing & demand", "🔎 Critical review", "📝 Survey"])

# ================================================================== EXECUTIVE SUMMARY
with tabs[0]:
    nps_b = kpi(res, "nps_buyers")
    cards = [
        ("Would buy", pct(kpi(res, "buy_rate")), f"±{pct(base['buy_rate_ci'])} (95% CI)", "good"),
        ("Most-probable segment", pct(kpi(res, "top_tier_share")), f"{pct(kpi(res, 'top_tier_buy'))} of them would buy", "good"),
        ("Median buyer spend", inr(kpi(res, "median_spend_buyers")), "per month (Q25)", "neutral"),
        ("Best price / drink", inr(kpi(res, "best_price")), "revenue-maximising (Q24)", "neutral"),
        ("Buyer NPS", f"{nps_b:.1f}", "promoters − detractors", "good" if nps_b > 0 else "bad"),
        ("Targeting power", f"{clf_best['cv_auc']:.2f} AUC", f"baseline 0.50 · accuracy {pct(clf_best['accuracy'])}", "good"),
    ]
    for row in (cards[:3], cards[3:]):
        for c, (l, v, s, t) in zip(st.columns(3), row):
            c.markdown(card(l, v, s, t), unsafe_allow_html=True)
        st.write("")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        section("Viability scorecard", "Rules of thumb chosen for this study, not industry standards.")
        rows = ""
        for s in res["scorecard"]:
            v = s["value"]
            val = pct(v) if s["fmt"] == "pct" else inr(v) if s["fmt"] == "inr" else (f"{v:.3f}" if 0 < v < 1 else f"{v:.1f}")
            pill = '<span class="pill pass">Pass</span>' if s["pass"] else '<span class="pill fail">Fail</span>'
            rows += f"<tr><td><b>{s['check']}</b><br><span style='color:#64748B;font-size:.8rem'>{s['measure']}</span></td><td>{val}</td><td>{s['rule']}</td><td>{pill}</td></tr>"
        st.markdown(f"<table class='sc'><tr><th>Check</th><th>Result</th><th>Bar</th><th>Status</th></tr>{rows}</table>", unsafe_allow_html=True)
    with right:
        section("What the data says")
        rules = pd.DataFrame(res["arm_rules"])
        big = rules.sort_values("support", ascending=False).iloc[0]
        best_seg = max(segs, key=lambda s: s["buy_rate"])
        rich_seg = max(segs, key=lambda s: s["expected_value"])
        g3 = res["gains"][2]
        bar = max(res["barriers"], key=lambda b: b["all"])
        st.markdown(f"""
1. **Demand is real but stated.** {pct(kpi(res, 'buy_rate'))} say yes — treat it as a ceiling, not a forecast.
2. **Buyers are findable.** Contacting the top 30% by model score reaches **{pct(g3['buyers_captured'], 0)}** of all buyers.
3. **Two segments carry it.** {NAMES[best_seg['segment']]} buy most often ({pct(best_seg['buy_rate'])}); {NAMES[rich_seg['segment']]} are worth most ({inr(rich_seg['expected_value'])}/person/month).
4. **Sell bundles.** {pct(big['confidence'], 0)} of buyers who want *{big['if_uses'].lower()}* also want *{big['then_uses'].lower()}*.
5. **Fix first:** *{bar['answer'].lower()}* is the top barrier ({pct(bar['all'])}) and buyers don't yet recommend it (NPS {nps_b:.1f}).
""")

    section("Is one outlet sustainable?", "Connects survey demand to break-even. Cost, catchment and conversion are assumptions — change them.")
    a, b, c = st.columns([1, 1, 1.4], gap="large")
    with a:
        fixed = st.number_input("Fixed cost per outlet per month (₹)", 100000, 3000000, 600000, step=50000,
                                help="Rent + staff + utilities + marketing for a metro café-bar. Placeholder.")
        margin = st.slider("Gross margin per drink", 0.30, 0.85, 0.60, 0.05)
    with b:
        catchment = st.number_input("Target-age people within reach of one outlet", 5000, 500000, 40000, step=5000,
                                    help="People aged 18–35 who could realistically visit. Placeholder.")
        conv = st.slider("Share of stated buyers who actually become regulars", 0.02, 1.0, 0.08, 0.02,
                         help="Stated intent overstates behaviour. Real regulars are a fraction of 'yes' answers.")
    spend = kpi(res, "median_spend_buyers")
    need = fixed / (spend * margin)
    expect = catchment * kpi(res, "buy_rate") * conv
    profit = expect * spend * margin - fixed
    with c:
        k1, k2 = st.columns(2)
        k1.markdown(card("Regulars needed", f"{need:,.0f}", f"at {inr(spend)}/month each", "neutral"), unsafe_allow_html=True)
        k2.markdown(card("Regulars expected", f"{expect:,.0f}", f"{pct(conv, 0)} of stated buyers", "good" if expect >= need else "bad"), unsafe_allow_html=True)
        st.markdown(card("Monthly operating result", ("+" if profit >= 0 else "−") + inr(abs(profit)),
                         "break-even reached" if profit >= 0 else "below break-even with these assumptions",
                         "good" if profit >= 0 else "bad"), unsafe_allow_html=True)

# ================================================================== DATA QUALITY
with tabs[1]:
    section("Data quality", "Real surveys are messy, so the synthetic data was made messy on purpose — then cleaned.")
    dq = {d["step"]: d["rows"] for d in res["data_quality"]}
    cols = st.columns(4)
    for col, (l, v, s) in zip(cols, [("Raw responses", f"{len(raw):,}", f"{raw.shape[1]} columns"),
                                      ("Straight-liners removed", f"{dq.get('Straight-liners removed', 0):,}", "same answer to all 7 ratings"),
                                      ("Missing cells filled", f"{dq.get('Missing answers filled', 0):,}", "median / most common answer"),
                                      ("Clean respondents", f"{kpi(res, 'respondents'):,}", "used by the models")]):
        col.markdown(card(l, v, s), unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        section("Cleaning steps")
        st.dataframe(pd.DataFrame(res["data_quality"]).rename(columns={"step": "Step", "rows": "Count", "note": "What it means"}),
                     hide_index=True, use_container_width=True)
    with b:
        section("Outliers (IQR rule on log scale)")
        od = pd.DataFrame(res["outliers"])
        od["share"] = od["share"].map(pct)
        for k in ["median", "max"]:
            od[k] = od[k].map(inr)
        st.dataframe(od.rename(columns={"variable": "Variable", "outliers": "Outliers", "share": "Share", "median": "Median", "max": "Max"}),
                     hide_index=True, use_container_width=True)
    bdf = raw[(raw.would_buy == 1) & (raw.monthly_spend > 0)]
    fig = px.box(bdf, x="monthly_spend", points="outliers", log_x=True, color_discrete_sequence=[TEAL],
                 labels={"monthly_spend": "Stated monthly spend on My Meltdown, ₹ (log scale)"},
                 title="A small group of extreme spenders sits far beyond everyone else")
    chart(fig, 300)
    with st.expander("Preview the raw dataset"):
        st.dataframe(raw.head(300), use_container_width=True, height=320)

# ================================================================== CLASSIFICATION
with tabs[2]:
    section("Classification — will this person buy?", "Predicts Q30 (would buy) from Q1–Q22 only, so the model cannot copy the answer.")
    g3 = res["gains"][2]
    takeaway(f"The best model ({clf_best['model'].lower()}) ranks people well (AUC {clf_best['cv_auc']:.2f}). Its accuracy is "
             f"{pct(clf_best['accuracy'])} against {pct(base['majority_acc'])} for always guessing 'won't buy'. In business terms: "
             f"the top 30% of people by score contain {pct(g3['buyers_captured'], 0)} of all buyers.")
    with st.expander("What is classification? (plain English)"):
        st.markdown("Classification sorts people into known groups — here **would buy** vs **would not buy**. Like a spam filter, it learns "
                    "from examples. We train on 75% of respondents and test on the other 25%, and repeat with 5-fold cross-validation.\n\n"
                    "**Accuracy** = share labelled correctly · **Precision** = of those flagged as buyers, how many are · "
                    "**Recall** = of real buyers, how many are found · **AUC** = ranking quality (0.5 coin flip, 1.0 perfect).")
    cm = pd.DataFrame(res["clf_models"])
    tbl = pd.concat([cm, pd.DataFrame([{"model": "Baseline: always 'won't buy'", "accuracy": base["majority_acc"], "precision": 0,
                                        "recall": 0, "f1": 0, "auc": 0.5, "cv_auc": 0.5, "best": 0}])], ignore_index=True)
    show = tbl[["model", "accuracy", "precision", "recall", "f1", "auc", "cv_auc"]].copy()
    for k in ["accuracy", "precision", "recall", "f1"]:
        show[k] = show[k].map(pct)
    show["model"] = np.where(tbl["best"] == 1, "⭐ " + tbl["model"], tbl["model"])
    st.dataframe(show.rename(columns={"model": "Model", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                                      "f1": "F1", "auc": "AUC (test)", "cv_auc": "AUC (5-fold)"}), hide_index=True, use_container_width=True)
    a, b = st.columns(2, gap="large")
    with a:
        gn = pd.DataFrame([{"people_share": 0, "buyers_captured": 0}] + res["gains"])
        fig = go.Figure()
        fig.add_scatter(x=gn.people_share, y=gn.buyers_captured, mode="lines+markers", name="Model ranking", line=dict(color=TEAL, width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random contact", line=dict(color=SLATE, dash="dash"))
        fig.update_layout(title="Cumulative gains: contact the top-scored people first", xaxis_title="Share of people contacted",
                          yaxis_title="Share of all buyers reached")
        fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(tickformat=".0%")
        chart(fig, 380)
    with b:
        th = pd.DataFrame(res["threshold"])
        fig = go.Figure()
        for k, nm, col in [("precision", "Precision", TEAL), ("recall", "Recall", CORAL), ("flagged", "Share flagged", SLATE)]:
            fig.add_scatter(x=th.threshold, y=th[k], name=nm, mode="lines", line=dict(color=col, width=3))
        fig.add_vline(x=0.5, line_dash="dot", line_color="#64748B", annotation_text="default 0.5")
        fig.update_layout(title="Choosing a cut-off: precision vs recall", xaxis_title="Probability cut-off for calling someone a buyer")
        fig.update_yaxes(tickformat=".0%")
        chart(fig, 380)
    a, b = st.columns(2, gap="large")
    with a:
        roc = pd.DataFrame(res["clf_roc"])
        fig = px.line(roc, x="fpr", y="tpr", color="model", color_discrete_sequence=[TEAL, CORAL, INDIGO],
                      labels={"fpr": "False positive rate", "tpr": "True positive rate"}, title="ROC curves: all three models perform alike")
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color=SLATE))
        chart(fig, 380)
    with b:
        cf = pd.DataFrame(res["clf_confusion"])
        mat = cf.pivot(index="actual", columns="predicted", values="count").loc[["Would buy", "Would not"], ["Would buy", "Would not"]]
        fig = px.imshow(mat, text_auto=True, color_continuous_scale=["#F0FDFA", TEAL], labels=dict(x="Predicted", y="Actual", color="People"),
                        title="Confusion matrix (best model, test set, cut-off 0.5)")
        fig.update_coloraxes(showscale=False)
        chart(fig, 380)
    dr = pd.DataFrame(res["clf_drivers"]).iloc[::-1]
    dr["Question"] = dr["question"] + " · " + dr["label"].str.slice(0, 64)
    fig = px.bar(dr, x="importance", y="Question", orientation="h", color="direction",
                 color_discrete_map={"Higher → more likely to buy": TEAL, "Higher → less likely to buy": ROSE,
                                     "Depends on answer": SLATE, "Weak / mixed": SLATE},
                 labels={"importance": "Drop in AUC when the answer is shuffled", "Question": ""},
                 title="Q17 (appeal of a hangover-free buzz) is by far the strongest predictor")
    fig.update_yaxes(categoryorder="array", categoryarray=dr["Question"].tolist())
    chart(fig, 470)

# ================================================================== SEGMENTATION
with tabs[3]:
    section("Segmentation — who are the customers?", "K-means on 14 habit and attitude answers (Q5–Q14, Q17–Q20), ranked by buy rate.")
    stab = res["clu_stability"]
    sil = pd.DataFrame(res["clu_silhouette"])
    chosen = sil[sil.chosen == 1].iloc[0]
    takeaway(f"Four segments. They overlap (silhouette {chosen.silhouette:.2f}) but are stable: re-running on random 80% samples "
             f"reproduces them (ARI {stab['ari_mean']:.2f}). Use them as personas for marketing; use classifier scores for targeting individuals.")
    with st.expander("What is clustering? (plain English)"):
        st.markdown("Clustering groups similar people **without being told the answer** — like friend groups forming at a party. "
                    "K-means places k 'average members' and assigns everyone to the nearest one, repeating until stable. "
                    "All answers are put on the same scale first. **Silhouette** (−1 to 1) says how cleanly groups separate; "
                    "**ARI** (0 to 1) says how reproducible they are.")
    tiers = pd.DataFrame(res["clu_tiers"])
    cols = st.columns(3)
    for col, t, tone in zip(cols, res["clu_tiers"], ["good", "neutral", "bad"]):
        col.markdown(card(t["tier"], pct(t["buy_rate"]), f"buy rate · {pct(t['share'])} of people · {pct(t['value_share'])} of expected revenue", tone),
                     unsafe_allow_html=True)
    st.write("")
    cols = [c for _ in range((len(segs) + 1) // 2) for c in st.columns(2)]
    for i, (col, s) in enumerate(zip(cols, segs)):
        nm = NAMES[s["segment"]]
        col.markdown(f"""<div class="persona" style="border-top:4px solid {SEG_COLORS[i]}">
<h4>{s['segment']} · {nm}</h4><div class="tier">{s['tier']}</div>
<div class="big">{pct(s['buy_rate'])}</div><div style="color:#64748B;font-size:.78rem">would buy</div>
<div class="desc">{PERSONA_DESC.get(nm, '')}</div>
<table><tr><td>Share of people</td><td class="r">{pct(s['share'])}</td></tr>
<tr><td>Median spend (buyers)</td><td class="r">{inr(s['median_spend_buyers'])}</td></tr>
<tr><td>Expected value / person</td><td class="r">{inr(s['expected_value'])}</td></tr>
<tr><td>Max price per drink</td><td class="r">{inr(s['median_max_price'])}</td></tr>
<tr><td>Drink weekly+</td><td class="r">{pct(s['weekly_drinkers'], 0)}</td></tr>
<tr><td>Favourite</td><td class="r">{s['top_format']}, {s['top_time'].lower()}</td></tr>
<tr><td>Top barrier</td><td class="r">{s['top_barrier']}</td></tr></table></div>""", unsafe_allow_html=True)
    st.write("")
    zp = pd.DataFrame(res["clu_zprofile"])
    zp["Segment"] = zp["segment"].map(seg_label)
    hm = zp.pivot(index="Segment", columns="feature", values="z")
    fig = px.imshow(hm, color_continuous_scale=[[0, "#B91C1C"], [0.5, "#FFFFFF"], [1, TEAL]], zmin=-1.2, zmax=1.2, aspect="auto",
                    text_auto=".1f", labels=dict(color="vs average"), title="Segment fingerprints: how each group differs from the average (standard deviations)")
    fig.update_xaxes(tickangle=-35, title="")
    fig.update_yaxes(title="")
    chart(fig, 360)
    a, b = st.columns(2, gap="large")
    with a:
        pts = pd.DataFrame(res["clu_points"])
        pts["Segment"] = pts["segment"].map(seg_label)
        fig = px.scatter(pts, x="pc1", y="pc2", color="Segment", opacity=0.55, color_discrete_sequence=SEG_COLORS,
                         category_orders={"Segment": sorted(pts["Segment"].unique())},
                         labels={"pc1": "Pattern 1 (PCA)", "pc2": "Pattern 2 (PCA)"}, title="Segment map: each dot is a person")
        fig.update_traces(marker=dict(size=5))
        chart(fig, 420)
    with b:
        sil["Choice"] = np.where(sil["chosen"] == 1, "Chosen", "Other")
        fig = px.bar(sil, x="k", y="silhouette", color="Choice", color_discrete_map={"Chosen": TEAL, "Other": "#CBD5E1"},
                     labels={"k": "Number of segments (k)", "silhouette": "Silhouette score"},
                     title="k = 4: best score once a mid tier is required")
        chart(fig, 420)

# ================================================================== SERVICE BUNDLES
with tabs[4]:
    section("Service bundles — association rule mining", "Apriori on Q23 (10 services) for people who said they would buy.")
    rules = pd.DataFrame(res["arm_rules"])
    big = rules.sort_values("support", ascending=False).iloc[0]
    strong = rules.sort_values("lift", ascending=False).iloc[0]
    takeaway(f"The most common bundle: buyers who want <i>{big['if_uses'].lower()}</i> also want <i>{big['then_uses'].lower()}</i> "
             f"{pct(big['confidence'], 0)} of the time ({pct(big['support'], 0)} of buyers). The strongest link (lift {strong['lift']:.2f}): "
             f"<i>{strong['if_uses'].lower()}</i> → <i>{strong['then_uses'].lower()}</i>.")
    with st.expander("What is association rule mining? (plain English)"):
        st.markdown("Market-basket analysis: 'people who buy bread and butter also buy jam'. A rule reads **if A and B, then C**.\n\n"
                    "**Support** = share of buyers using every service in the rule · **Confidence** = of those using the 'if' services, "
                    "share also using the 'then' service · **Lift** = how many times likelier the 'then' service becomes (>1 = real link).")
    a, b = st.columns([1, 1.25], gap="large")
    with a:
        items = pd.DataFrame(res["arm_items"]).sort_values("share_buyers")
        fig = go.Figure()
        fig.add_bar(y=items["service"], x=items["share_all"], orientation="h", name="Everyone", marker_color="#CBD5E1")
        fig.add_bar(y=items["service"], x=items["share_buyers"], orientation="h", name="Buyers", marker_color=TEAL)
        fig.update_layout(barmode="group", title="Night kavatails are the most wanted service")
        fig.update_xaxes(tickformat=".0%")
        chart(fig, 520)
    with b:
        sc = rules.copy()
        sc["label"] = sc["if_uses"] + " → " + sc["then_uses"]
        fig = px.scatter(sc, x="support", y="confidence", size="lift", color="lift", hover_name="label",
                         color_continuous_scale=["#CCFBF1", TEAL], size_max=22,
                         labels={"support": "Support (bundle size)", "confidence": "Confidence"},
                         title="Every rule: big bundles (right) vs strong links (large, dark)")
        fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(tickformat=".0%")
        chart(fig, 520)
    f1, f2, f3 = st.columns(3)
    min_conf = f1.slider("Minimum confidence", 0.55, 0.95, 0.55, 0.05)
    min_lift = f2.slider("Minimum lift", 1.2, 2.0, 1.2, 0.05)
    pairs = f3.toggle("Only 'two services → a third'", value=False)
    rl = rules[(rules.confidence >= min_conf) & (rules.lift >= min_lift)]
    if pairs:
        rl = rl[rl.antecedent_size == 2]
    show = rl[["if_uses", "then_uses", "confidence", "lift", "support", "respondents"]].copy()
    st.dataframe(show.rename(columns={"if_uses": "If they use", "then_uses": "They also use", "confidence": "Confidence",
                                      "lift": "Lift", "support": "Support", "respondents": "Buyers"}), hide_index=True, use_container_width=True,
                 column_config={"Confidence": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                                "Support": st.column_config.NumberColumn(format="%.3f"), "Lift": st.column_config.NumberColumn(format="%.2f")})
    st.caption(f"{len(rl)} of {len(rules)} rules shown. Mining cut-offs: support ≥ 5%, confidence ≥ 55%, lift ≥ 1.2.")

# ================================================================== SPEND
with tabs[5]:
    section("Spend model — regression", "Predicts Q25 (monthly spend) for buyers from Q1–Q22, on a log scale.")
    rb = res["reg_baseline"]
    best_r = next(m for m in res["reg_models"] if m["best"] == 1)
    left_in = [m for m in res["reg_models"] if m["outliers"] == "Left in"]
    takeaway(f"Handling outliers is essential: left in, R² falls to {min(m['r2'] for m in left_in):.1f}. Capped, the best model explains "
             f"R² = {best_r['r2']:.2f} of spend (log scale) with a median error of {inr(best_r['medae'])}, versus {inr(rb['medae'])} "
             f"for simply guessing the median. Good for ranking spenders; too rough for individual forecasts.")
    with st.expander("What is regression? (plain English)"):
        st.markdown("Regression predicts a **number** — like estimating a flat's price from its size and location. "
                    "**R²** = share of the variation explained (0 = no better than the average, negative = worse) · "
                    "**Mean / median error** = how far off a typical prediction is, in rupees.")
    rm = pd.DataFrame(res["reg_models"])
    rm = pd.concat([rm, pd.DataFrame([{"model": "Baseline: always guess the median", "outliers": "—", "r2": rb["r2"], "mae": rb["mae"],
                                        "medae": rb["medae"], "best": 0}])], ignore_index=True)
    rs = rm[["model", "outliers", "r2", "mae", "medae"]].copy()
    rs["model"] = np.where(rm["best"] == 1, "⭐ " + rm["model"], rm["model"])
    rs["mae"] = rs["mae"].map(inr); rs["medae"] = rs["medae"].map(inr)
    st.dataframe(rs.rename(columns={"model": "Model", "outliers": "Outliers", "r2": "R² (log spend)", "mae": "Mean error",
                                    "medae": "Median error"}), hide_index=True, use_container_width=True)
    a, b = st.columns(2, gap="large")
    with a:
        rp = pd.DataFrame(res["reg_points"])
        rp = rp[(rp.actual > 0) & (rp.predicted > 0)]
        fig = px.scatter(rp, x="actual", y="predicted", log_x=True, log_y=True, opacity=0.6, color_discrete_sequence=[TEAL],
                         labels={"actual": "Actual monthly spend (₹)", "predicted": "Predicted (₹)"},
                         title="Predicted vs actual spend (test buyers)")
        lo, hi = 50, max(rp.actual.max(), rp.predicted.max())
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color=SLATE))
        chart(fig, 420)
    with b:
        rd = pd.DataFrame(res["reg_drivers"]).iloc[::-1]
        rd["Question"] = rd["question"] + " · " + rd["label"].str.slice(0, 52)
        fig = px.bar(rd, x="importance", y="Question", orientation="h", color_discrete_sequence=[CORAL],
                     labels={"importance": "Drop in R² when shuffled", "Question": ""}, title="Today's café and bar regulars spend the most")
        chart(fig, 420)
    hist = pd.DataFrame(res["spend_hist"])
    hist["tail"] = np.where(hist["lo"] >= 5000, "Long tail (₹5,000+)", "Typical")
    fig = px.bar(hist, x="bin", y="count", color="tail", color_discrete_map={"Typical": TEAL, "Long tail (₹5,000+)": ROSE},
                 labels={"bin": "Monthly spend band", "count": "Buyers"}, title="Most buyers would spend ₹250–₹1,500 a month")
    chart(fig, 340)

# ================================================================== PRICING & DEMAND
with tabs[6]:
    section("Pricing & demand", "Q24 (max price), Q27 (recommend), Q28 (barrier), Q29 (channel).")
    pc = pd.DataFrame(res["price_curve"])
    bp = kpi(res, "best_price")
    takeaway(f"Revenue peaks at {inr(bp)} a drink, where {pct(pc[pc.price == bp].share_buyers.iloc[0], 0)} of buyers are still willing. "
             f"Above ₹250 willingness drops fast. No single barrier dominates, so taste, availability, safety and price all need answers.")
    fig = go.Figure()
    fig.add_bar(x=pc.price, y=pc.revenue_index, name="Revenue index (price × share of buyers)", marker_color="#CCFBF1", yaxis="y2")
    fig.add_scatter(x=pc.price, y=pc.share_buyers, name="Buyers willing", mode="lines+markers", line=dict(color=TEAL, width=3))
    fig.add_scatter(x=pc.price, y=pc.share_all, name="Everyone willing", mode="lines", line=dict(color=SLATE, width=2, dash="dot"))
    fig.add_vline(x=bp, line_dash="dash", line_color=CORAL, annotation_text=f"Revenue-best ₹{bp}")
    fig.update_layout(title="Willingness to pay per drink", xaxis_title="Price per drink (₹)",
                      yaxis=dict(tickformat=".0%", title="Share willing"),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Revenue index"))
    chart(fig, 400)
    a, b, c = st.columns(3, gap="large")
    for col, key, title in [(a, "barriers", "What would stop them"), (b, "channels", "Where they want to buy")]:
        d = pd.DataFrame(res[key]).sort_values("all")
        fig = go.Figure()
        fig.add_bar(y=d.answer, x=d["all"], orientation="h", name="Everyone", marker_color="#CBD5E1")
        fig.add_bar(y=d.answer, x=d["buyers"], orientation="h", name="Buyers", marker_color=TEAL)
        fig.update_layout(barmode="group", title=title)
        fig.update_xaxes(tickformat=".0%")
        with col:
            chart(fig, 400)
    with c:
        cl = res["clean"]
        def nps_mix(s):
            return [(s <= 6).mean(), ((s >= 7) & (s <= 8)).mean(), (s >= 9).mean()]
        mix = pd.DataFrame({"Group": ["Everyone", "Buyers"], "mix": [nps_mix(cl.nps_score), nps_mix(cl[cl.would_buy == 1].nps_score)]})
        fig = go.Figure()
        for i, (nm, colr) in enumerate([("Detractors (0–6)", ROSE), ("Passives (7–8)", "#CBD5E1"), ("Promoters (9–10)", TEAL)]):
            fig.add_bar(y=mix.Group, x=[m[i] for m in mix["mix"]], orientation="h", name=nm, marker_color=colr)
        fig.update_layout(barmode="stack", title="Would they recommend it? (Q27)")
        fig.update_xaxes(tickformat=".0%")
        chart(fig, 400)

# ================================================================== CRITICAL REVIEW
with tabs[7]:
    section("Critical review — how far can these results be trusted?",
            "Each point gives the evidence from this run, how much it matters, and what to do about it.")
    g1, g3 = res["gains"][0], res["gains"][2]
    rb = res["reg_baseline"]
    best_r = next(m for m in res["reg_models"] if m["best"] == 1)
    tiers = {t["tier"]: t for t in res["clu_tiers"]}
    arm_pairs = kpi(res, "pair_rules")
    items = [
        ("high", "The data is synthetic, so the conclusions are partly built in",
         "Answers were generated from assumed relationships (e.g. open-minded people like the concept more). The models then rediscover those assumptions.",
         "The dashboard proves the survey and pipeline work end to end. Run the 30 questions on 300–500 real people and upload the file in the sidebar."),
        ("high", "Stated intent is not behaviour",
         f"{pct(base['buy_rate'])} say they would buy. The ±{pct(base['buy_rate_ci'])} interval is about sampling only; it ignores the gap between saying and doing.",
         "Treat the buy rate as a ceiling. The sustainability calculator applies a conversion assumption; validate it with pre-orders or a pop-up pilot."),
        ("med", "Classifier accuracy looks better than it is",
         f"Accuracy {pct(clf_best['accuracy'])} vs {pct(base['majority_acc'])} for always guessing 'won't buy'. AUC {clf_best['cv_auc']:.2f} is useful but moderate; "
         f"3% of answers were flipped as noise, which caps any model.",
         f"Judge it by ranking instead: the top 10% by score buy at {pct(g1['buy_rate'], 0)} (lift {g1['lift']:.1f}×) and the top 30% hold {pct(g3['buyers_captured'], 0)} of buyers."),
        ("med", "The default 0.5 cut-off misses many buyers",
         f"At 0.5 recall is {pct(clf_best['recall'])}: about {pct(1 - clf_best['recall'], 0)} of real buyers are not flagged.",
         "Pick the cut-off from campaign economics (see the precision–recall chart): cheap outreach → lower cut-off; expensive → higher."),
        ("med", "Segments overlap",
         f"Silhouette {chosen.silhouette:.2f} (weak separation). k = 4 was chosen to allow a mid tier although k = 2 scored slightly higher. "
         f"Tier buy rates: {pct(tiers['Most probable']['buy_rate'])} vs {pct(tiers['Mid probable']['buy_rate'])} vs {pct(tiers['Least probable']['buy_rate'])}.",
         f"Segments are stable (ARI {stab['ari_mean']:.2f}), so use them as personas for messaging and menus. For who-to-target decisions use classifier scores."),
        ("low", "Some strong rules rest on small groups",
         "The highest-lift rules involve the monthly membership, which only about a quarter of buyers want, so their support is 5–13%.",
         f"Prioritise high-support bundles (the night-out bundle covers about a third of buyers). {arm_pairs} 'two-services → third' rules exist; validate the top 3 with real menus."),
        ("med", "Spend is hard to predict",
         f"R² {best_r['r2']:.2f} on log spend; median error {inr(best_r['medae'])} vs {inr(rb['medae'])} for a naive median guess. With outliers left in, R² goes negative.",
         "Use the model to rank likely high spenders, and plan revenue on the median buyer, not the mean."),
        ("med", "Break-even rests on placeholders",
         "Fixed costs, margin, catchment size and conversion are assumptions, not quotes or market data.",
         "Replace them with real rent and staff quotes and a catchment estimate for the chosen location; the calculator updates live."),
        ("med", "Survey design biases",
         "Self-reported money answers are noisy; people over-report healthy choices; an online survey over-samples young, urban, curious people.",
         "Use quotas by age and city tier, randomise option order, add an attention-check question, and pilot the survey on 20 people first."),
        ("high", "Regulation is outside the data",
         "Kava has no FSSAI product standard; it likely needs novel-food approval, and 'tea' cannot be used on labels.",
         "Confirm the regulatory route before any spend. If approval is unlikely, the whole analysis is moot."),
    ]
    sev = {"high": ("High impact", "high"), "med": ("Medium impact", "med"), "low": ("Low impact", "low")}
    for lvl, title, ev, act in items:
        lab, cls = sev[lvl]
        st.markdown(f"""<div class="review"><h4><span class="pill {cls}">{lab}</span>&nbsp; {title}</h4>
<p><b>Evidence:</b> {ev}</p><p><b>What to do:</b> {act}</p></div>""", unsafe_allow_html=True)

# ================================================================== SURVEY
with tabs[8]:
    section("The 30-question survey", "Show a two-line description of My Meltdown before Section C. 'Feeds' = the model(s) each answer is used in.")
    st.caption("CLF = classification feature · CLF-T = classification target · CLU = clustering · ARM = association rules · "
               "REG = regression feature · REG-T = regression target · PROF = segment profiling · BIZ = business figure")
    sv = pd.DataFrame(res["survey"])[["q", "section", "question", "type", "options", "feeds"]]
    st.dataframe(sv.rename(columns={"q": "#", "section": "Section", "question": "Question", "type": "Answer type",
                                    "options": "Options", "feeds": "Feeds"}), hide_index=True, use_container_width=True, height=640)

st.markdown('<div class="foot">My Meltdown · idea validation dashboard · Python, scikit-learn, Plotly, Streamlit · '
            'Synthetic data unless a real survey file is uploaded.</div>', unsafe_allow_html=True)
