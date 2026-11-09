"""My Meltdown analysis pipeline: cleaning + classification, clustering,
association rules and regression. run(raw_df) returns every result table.
Works on real survey exports too, as long as column names match survey.py.
"""
import sys, os, json, itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, r2_score, mean_absolute_error, mean_squared_error, silhouette_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from survey import QUESTIONS, SERVICES

def run(raw, SEED=7):
    """Run the full pipeline on a raw survey DataFrame. Returns a dict of results."""
    out = {}
    df = raw.copy()

    def save(name, rows):
        out[name] = rows

    def r(x, n=3):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)

    # =============== 1. CLEANING ===============
    LIK = ["want_cut_alcohol", "wellness", "stress", "buzz_appeal", "safety_concern", "novelty_seeking", "influencer_effect"]
    quality = []
    n0 = len(df)
    quality.append({"step": "Raw responses", "rows": n0, "note": "All submitted responses"})

    missing_cells = int(df.drop(columns=[c for c in ["injected_issue"] if c in df]).isna().sum().sum())
    straight = df[LIK].std(axis=1) == 0
    df = df[~straight].copy()
    quality.append({"step": "Straight-liners removed", "rows": int(straight.sum()), "note": "Same answer on all 7 agree/disagree questions"})

    imp_sm = df.social_media_hrs > 18
    imp_cv = df.cafe_visits_month > 60
    zero_inc = df.monthly_income <= 0
    df.loc[imp_sm, "social_media_hrs"] = np.nan
    df.loc[imp_cv, "cafe_visits_month"] = np.nan
    df.loc[zero_inc, "monthly_income"] = np.nan
    quality.append({"step": "Impossible values blanked", "rows": int(imp_sm.sum() + imp_cv.sum() + zero_inc.sum()),
                    "note": ">18 h/day social media, >60 café visits/month, ₹0 income"})

    def iqr_flags(s, log=True):
        x = np.log1p(s) if log else s
        q1, q3 = x.quantile(.25), x.quantile(.75)
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        return (x < lo) | (x > hi)

    out_cols = {"monthly_income": "Income", "cafe_spend_visit": "Café spend per visit", "nightout_spend": "Night-out spend",
                "monthly_spend": "Monthly spend on My Meltdown (buyers)"}
    outlier_rows = []
    for c, lab in out_cols.items():
        s = df.loc[df.would_buy == 1, c] if c == "monthly_spend" else df[c]
        s = s[s > 0] if c == "monthly_spend" else s
        fl = iqr_flags(s.dropna())
        outlier_rows.append({"variable": lab, "outliers": int(fl.sum()), "share": r(fl.mean(), 4),
                             "median": r(s.median(), 0), "max": r(s.max(), 0)})
    total_out = sum(o["outliers"] for o in outlier_rows)
    quality.append({"step": "Outliers flagged (IQR rule, log scale)", "rows": total_out, "note": "Kept; features capped at 1st/99th pct, spend target capped at the IQR fence"})
    quality.append({"step": "PLACEHOLDER_INCONS"})
    quality.append({"step": "Missing answers filled", "rows": missing_cells, "note": "Median for numbers, most common answer for choices (cells)"})
    quality.append({"step": "Clean responses used", "rows": len(df), "note": "Rows going into the four models"})
    save("outliers", outlier_rows)

    # impute
    NUM = ["monthly_income", "cafe_visits_month", "bar_visits_month", "cafe_spend_visit", "nightout_spend", "social_media_hrs", "max_price_drink"]
    for c in NUM:
        df[c] = df[c].fillna(df[c].median())
    for c in ["kava_awareness"]:
        df[c] = df[c].fillna(df[c].mode()[0])
    spend_raw = df.monthly_spend.copy()
    for c in ["monthly_income", "cafe_spend_visit", "nightout_spend", "cafe_visits_month"]:
        lo, hi = df[c].quantile([.01, .99])
        df[c] = df[c].clip(lo, hi)

    # =============== 2. FEATURES (Q1-Q22) ===============
    ORD = {
        "age_group": ["18-20", "21-24", "25-29", "30-35", "36-45"],
        "city_tier": ["Tier 3", "Tier 2", "Tier 1", "Metro"],
        "alcohol_freq": ["Never", "Rarely", "Monthly", "Weekly", "2+ per week"],
        "kava_awareness": ["Never heard", "Heard of it", "Tried it"],
    }
    X = pd.DataFrame(index=df.index)
    for c, levels in ORD.items():
        X[c] = df[c].map({v: i for i, v in enumerate(levels)})
    for c in ["monthly_income", "cafe_spend_visit", "nightout_spend"]:
        X[c] = np.log1p(df[c])
    for c in ["cafe_visits_month", "bar_visits_month", "social_media_hrs", "tried_na_alt"] + LIK:
        X[c] = df[c]
    ONEHOT = ["gender", "occupation", "preferred_time", "preferred_format"]
    X = pd.concat([X, pd.get_dummies(df[ONEHOT], prefix=ONEHOT, dtype=int)], axis=1)
    X.columns = [str(c) for c in X.columns]
    QLABEL = {q[5]: (q[0], q[2]) for q in QUESTIONS}

    def group_of(col):
        for g in ONEHOT:
            if col.startswith(g + "_"):
                return g
        return col

    y = df.would_buy.values

    # =============== 3. CLASSIFICATION ===============
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=.5)),
        "Random forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=1),
        "Gradient boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=.05, max_iter=300, random_state=SEED),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    clf_rows, roc_rows, fitted = [], [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        pred = (p >= .5).astype(int)
        cv_auc = cross_val_score(m, X, y, cv=cv, scoring="roc_auc")
        clf_rows.append({"model": name, "accuracy": r(accuracy_score(yte, pred)), "precision": r(precision_score(yte, pred)),
                         "recall": r(recall_score(yte, pred)), "f1": r(f1_score(yte, pred)), "auc": r(roc_auc_score(yte, p)),
                         "cv_auc": r(cv_auc.mean()), "cv_auc_sd": r(cv_auc.std())})
        fpr, tpr, _ = roc_curve(yte, p)
        keep = np.unique(np.linspace(0, len(fpr) - 1, 60).astype(int))
        roc_rows += [{"model": name, "fpr": r(fpr[i]), "tpr": r(tpr[i])} for i in keep]
        fitted[name] = m
    best_clf = max(clf_rows, key=lambda d: d["cv_auc"])["model"]
    for row in clf_rows:
        row["best"] = int(row["model"] == best_clf)
    save("clf_models", clf_rows)
    save("clf_roc", roc_rows)
    bm = fitted[best_clf]
    pb = (bm.predict_proba(Xte)[:, 1] >= .5).astype(int)
    cm = confusion_matrix(yte, pb)
    save("clf_confusion", [
        {"actual": "Would buy", "predicted": "Would buy", "count": int(cm[1, 1])},
        {"actual": "Would buy", "predicted": "Would not", "count": int(cm[1, 0])},
        {"actual": "Would not", "predicted": "Would buy", "count": int(cm[0, 1])},
        {"actual": "Would not", "predicted": "Would not", "count": int(cm[0, 0])},
    ])

    def grouped_importance(model, Xv, yv, scoring):
        pi = permutation_importance(model, Xv, yv, n_repeats=5, random_state=SEED, scoring=scoring, n_jobs=1)
        s = pd.Series(pi.importances_mean, index=Xv.columns).groupby(group_of).sum()
        return s.sort_values(ascending=False)

    imp = grouped_importance(bm, Xte, yte, "roc_auc")
    corr_target = pd.Series(y, index=X.index)
    drivers = []
    for var, v in imp.head(12).items():
        qn, qt = QLABEL.get(var, (None, var))
        if var in X.columns:
            rho = pd.Series(X[var]).corr(corr_target, method="spearman")
            direction = "Higher → more likely to buy" if rho > 0.03 else ("Higher → less likely to buy" if rho < -0.03 else "Weak / mixed")
        else:
            direction = "Depends on answer"
        drivers.append({"variable": var, "question": f"Q{qn}" if qn else "", "label": qt, "importance": r(v, 4), "direction": direction})
    save("clf_drivers", drivers)
    # out-of-fold buy probability for every respondent (used in segments)
    df["p_buy"] = cross_val_predict(models[best_clf], X, y, cv=cv, method="predict_proba")[:, 1]

    # =============== 4. CLUSTERING ===============
    CLU = ["monthly_income", "alcohol_freq", "want_cut_alcohol", "cafe_visits_month", "bar_visits_month", "cafe_spend_visit",
           "nightout_spend", "wellness", "stress", "social_media_hrs", "buzz_appeal", "safety_concern", "novelty_seeking", "influencer_effect"]
    Z = StandardScaler().fit_transform(X[CLU])
    sil_rows = []
    samp = np.random.default_rng(SEED).choice(len(Z), min(2500, len(Z)), replace=False)
    km_by_k = {}
    for k in range(2, 8):
        km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
        km_by_k[k] = km
        sil_rows.append({"k": k, "silhouette": r(silhouette_score(Z[samp], km.labels_[samp]), 4), "inertia": r(km.inertia_, 0)})
    # pick best silhouette among k>=4 so we can split into most / mid / least probable with detail
    cand = [s for s in sil_rows if s["k"] >= 4]
    K = max(cand, key=lambda s: s["silhouette"])["k"]
    for s in sil_rows:
        s["chosen"] = int(s["k"] == K)
    save("clu_silhouette", sil_rows)
    km = km_by_k[K]
    df["cluster"] = km.labels_

    prof = []
    for c, g in df.groupby("cluster"):
        buyers = g[g.would_buy == 1]
        prof.append({"cluster": int(c), "size": len(g), "share": len(g) / len(df), "buy_rate": g.would_buy.mean(),
                     "avg_p_buy": g.p_buy.mean(),
                     "median_spend_buyers": float(np.median(spend_raw.loc[buyers.index])) if len(buyers) else 0,
                     "median_max_price": g.max_price_drink.median(),
                     "median_income": g.monthly_income.median(),
                     "nps": (g.nps_score >= 9).mean() - (g.nps_score <= 6).mean(),
                     "alcohol_weekly_plus": g.alcohol_freq.isin(["Weekly", "2+ per week"]).mean(),
                     "buzz_appeal": g.buzz_appeal.mean(), "safety_concern": g.safety_concern.mean(),
                     "novelty": g.novelty_seeking.mean(), "wellness": g.wellness.mean(),
                     "cafe_visits": g.cafe_visits_month.median(), "bar_visits": g.bar_visits_month.median(),
                     "top_age": g.age_group.mode()[0], "top_city": g.city_tier.mode()[0], "top_occupation": g.occupation.mode()[0],
                     "top_time": g.preferred_time.mode()[0], "top_format": g.preferred_format.mode()[0],
                     "top_channel": g.preferred_channel.mode()[0], "top_barrier": g.main_barrier.mode()[0]})
    prof = sorted(prof, key=lambda d: -d["buy_rate"])
    n = len(prof)
    for i, p in enumerate(prof):
        p["tier"] = "Most probable" if i == 0 else ("Least probable" if i == n - 1 else "Mid probable")
        p["rank"] = i + 1

    pca = PCA(2, random_state=SEED).fit(Z)
    P = pca.transform(Z)
    pts = np.random.default_rng(1).choice(len(P), min(1400, len(P)), replace=False)
    tier_of = {p["cluster"]: p["tier"] for p in prof}
    save("clu_points", [{"id": df.respondent_id.iloc[i], "pc1": r(P[i, 0]), "pc2": r(P[i, 1]),
                         "cluster": int(df.cluster.iloc[i]), "tier": tier_of[int(df.cluster.iloc[i])]} for i in pts])
    df["tier"] = df.cluster.map(tier_of)

    # =============== 5. ASSOCIATION RULES (Apriori, buyers) ===============
    items = [s for s, _ in SERVICES]
    names = dict(SERVICES)
    B = df.loc[df.would_buy == 1, items].astype(bool)
    nB = len(B)
    def support(sset):
        return B[list(sset)].all(axis=1).mean()
    MIN_SUP, MIN_CONF, MIN_LIFT = .05, .55, 1.2
    freq = {}
    for k in (1, 2, 3):
        for combo in itertools.combinations(items, k):
            if k > 1 and any(frozenset(sub) not in freq for sub in itertools.combinations(combo, k - 1)):
                continue
            s = support(combo)
            if s >= MIN_SUP:
                freq[frozenset(combo)] = s
    rules = []
    for iset, s in freq.items():
        if len(iset) < 2:
            continue
        for k in range(1, len(iset)):
            for ante in itertools.combinations(sorted(iset, key=items.index), k):
                ante = frozenset(ante)
                cons = iset - ante
                if len(cons) != 1:
                    continue
                conf = s / freq[ante]
                lift = conf / freq[cons]
                if conf >= MIN_CONF and lift >= MIN_LIFT:
                    a = sorted(ante, key=items.index)
                    c = next(iter(cons))
                    rules.append({"rule_id": "+".join(a) + "→" + c, "if_uses": " + ".join(names[x] for x in a),
                                  "then_uses": names[c], "antecedent_size": len(a),
                                  "support": r(s, 4), "confidence": r(conf, 4), "lift": r(lift, 3),
                                  "respondents": int(round(s * nB))})
    rules = sorted(rules, key=lambda d: (-d["lift"], -d["confidence"]))
    save("arm_rules", rules[:30])
    save("arm_items", [{"code": s, "service": names[s], "share_buyers": r(B[s].mean(), 4),
                        "share_all": r(df[s].mean(), 4)} for s in items])
    arm_summary = {"buyers": nB, "frequent_itemsets": len(freq), "rules_found": len(rules),
                   "pair_rules": sum(1 for x in rules if x["antecedent_size"] == 2)}

    # =============== 6. REGRESSION (monthly spend, buyers) ===============
    bi = df.index[(df.would_buy == 1) & (spend_raw > 0)]
    n_incons = int(((df.would_buy == 1) & (spend_raw <= 0)).sum())
    Xb, yb_raw = X.loc[bi], spend_raw.loc[bi].values
    _pos = np.log1p(yb_raw)
    _q1, _q3 = np.quantile(_pos, [.25, .75])
    cap = float(np.expm1(_q3 + 1.5 * (_q3 - _q1)))  # IQR upper fence on log spend
    yb = np.clip(yb_raw, 0, cap)
    idx_tr, idx_te = train_test_split(np.arange(len(bi)), test_size=.25, random_state=SEED)
    y_eval = yb[idx_te]
    reg_rows, reg_fit = [], {}
    def evaluate(name, pred, treat):
        pred = np.clip(pred, 0, None)
        reg_rows.append({"model": name, "outliers": treat, "r2": r(r2_score(np.log1p(y_eval), np.log1p(pred))),
                         "mae": r(mean_absolute_error(y_eval, pred), 0),
                         "medae": r(np.median(np.abs(y_eval - pred)), 0)})
    lin_raw = LinearRegression().fit(Xb.iloc[idx_tr], yb_raw[idx_tr])
    evaluate("Linear regression", lin_raw.predict(Xb.iloc[idx_te]), "Left in")
    rf_raw = RandomForestRegressor(200, min_samples_leaf=5, random_state=SEED, n_jobs=1).fit(Xb.iloc[idx_tr], yb_raw[idx_tr])
    evaluate("Random forest", rf_raw.predict(Xb.iloc[idx_te]), "Left in")
    class LogModel:
        def __init__(self, m): self.m = m
        def fit(self, X, y): self.m.fit(X, np.log1p(y)); return self
        def predict(self, X): return np.expm1(self.m.predict(X))
        def score(self, X, y): return r2_score(np.log1p(y), np.log1p(np.clip(self.predict(X), 0, None)))
    lin = LogModel(make_pipeline(StandardScaler(), LinearRegression())).fit(Xb.iloc[idx_tr], yb[idx_tr])
    evaluate("Linear regression (log)", lin.predict(Xb.iloc[idx_te]), "Capped at IQR fence"); reg_fit["Linear regression (log)"] = lin
    rf = LogModel(RandomForestRegressor(200, min_samples_leaf=8, random_state=SEED, n_jobs=1)).fit(Xb.iloc[idx_tr], yb[idx_tr])
    evaluate("Random forest (log)", rf.predict(Xb.iloc[idx_te]), "Capped at IQR fence"); reg_fit["Random forest (log)"] = rf
    gb = LogModel(HistGradientBoostingRegressor(max_depth=3, learning_rate=.04, max_iter=300, random_state=SEED)).fit(Xb.iloc[idx_tr], yb[idx_tr])
    evaluate("Gradient boosting (log)", gb.predict(Xb.iloc[idx_te]), "Capped at IQR fence"); reg_fit["Gradient boosting (log)"] = gb
    clean_rows = [x for x in reg_rows if x["outliers"] != "Left in"]
    best_reg = max(clean_rows, key=lambda d: d["r2"])["model"]
    for x in reg_rows:
        x["best"] = int(x["model"] == best_reg and x["outliers"] != "Left in")
        x["id"] = f'{x["model"]} | {x["outliers"]}'
    save("reg_models", reg_rows)
    rbm = reg_fit[best_reg]
    pred_te = rbm.predict(Xb.iloc[idx_te])
    save("reg_points", [{"id": df.respondent_id.loc[bi[i]], "actual": r(y_eval[j], 0), "predicted": r(max(pred_te[j], 0), 0)}
                        for j, i in enumerate(idx_te)])
    rimp = grouped_importance(rbm, Xb.iloc[idx_te], y_eval, None)
    rdr = []
    for var, v in rimp.head(10).items():
        qn, qt = QLABEL.get(var, (None, var))
        rdr.append({"variable": var, "question": f"Q{qn}" if qn else "", "label": qt, "importance": r(v, 4)})
    save("reg_drivers", rdr)
    edges = list(range(0, 5001, 250))
    hist = []
    for lo in edges[:-1]:
        hist.append({"bin": f"₹{lo:,}–{lo + 250:,}", "lo": lo, "count": int(((yb_raw >= lo) & (yb_raw < lo + 250)).sum())})
    hist.append({"bin": "₹5,000+", "lo": 5000, "count": int((yb_raw >= 5000).sum())})
    save("spend_hist", hist)
    # expected monthly value per respondent = P(buy) x predicted spend
    df["pred_spend"] = np.clip(rbm.predict(X), 0, None)
    df["exp_value"] = df.p_buy * df.pred_spend

    # segment table
    seg_rows = []
    for p in prof:
        g = df[df.cluster == p["cluster"]]
        seg_rows.append({"segment": f'S{p["rank"]}', "tier": p["tier"], "respondents": p["size"], "share": r(p["share"], 4),
                         "buy_rate": r(p["buy_rate"], 4), "median_spend_buyers": r(p["median_spend_buyers"], 0),
                         "expected_value": r(g.exp_value.mean(), 0), "median_max_price": r(p["median_max_price"], 0),
                         "nps": r(p["nps"] * 100, 0), "median_income": r(p["median_income"], 0),
                         "weekly_drinkers": r(p["alcohol_weekly_plus"], 4), "buzz_appeal": r(p["buzz_appeal"], 2),
                         "safety_concern": r(p["safety_concern"], 2), "novelty": r(p["novelty"], 2), "wellness": r(p["wellness"], 2),
                         "cafe_visits": r(p["cafe_visits"], 0), "bar_visits": r(p["bar_visits"], 0),
                         "top_age": p["top_age"], "top_city": p["top_city"], "top_occupation": p["top_occupation"],
                         "top_time": p["top_time"], "top_format": p["top_format"], "top_channel": p["top_channel"],
                         "top_barrier": p["top_barrier"], "cluster": p["cluster"]})
    save("clu_segments", seg_rows)
    tier_rows = []
    for t in ["Most probable", "Mid probable", "Least probable"]:
        g = df[df.tier == t]
        tier_rows.append({"tier": t, "respondents": len(g), "share": r(len(g) / len(df), 4), "buy_rate": r(g.would_buy.mean(), 4),
                          "expected_value": r(g.exp_value.mean(), 0),
                          "value_share": r(g.exp_value.sum() / df.exp_value.sum(), 4)})
    save("clu_tiers", tier_rows)
    # rename points' cluster to segment code
    segcode = {p["cluster"]: f'S{p["rank"]}' for p in prof}
    pts_rows = out["clu_points"]
    for p in pts_rows:
        p["segment"] = segcode[p["cluster"]]
    save("clu_points", pts_rows)

    for q in quality:
        if q["step"] == "PLACEHOLDER_INCONS":
            q.update({"step": "Inconsistent answers left out of spend model", "rows": n_incons, "note": "Said they would buy (Q30) but ₹0 monthly spend (Q25)"})
    save("data_quality", quality)
    # =============== 7. BUSINESS KPIs ===============
    price_rows = []
    for pr in range(100, 501, 25):
        sa = (df.max_price_drink >= pr).mean()
        sb = (df.loc[df.would_buy == 1, "max_price_drink"] >= pr).mean()
        price_rows.append({"price": pr, "share_all": r(sa, 4), "share_buyers": r(sb, 4), "revenue_index": r(pr * sb, 1)})
    best_price = max(price_rows, key=lambda d: d["revenue_index"])["price"]
    for p in price_rows:
        p["best"] = int(p["price"] == best_price)
    save("price_curve", price_rows)
    def counts(col):
        rows = []
        for v in df[col].value_counts().index:
            rows.append({"answer": v, "all": r((df[col] == v).mean(), 4),
                         "buyers": r((df.loc[df.would_buy == 1, col] == v).mean(), 4)})
        return rows
    save("barriers", counts("main_barrier"))
    save("channels", counts("preferred_channel"))

    buyers = df[df.would_buy == 1]
    def nps(s):
        return ((s >= 9).mean() - (s <= 6).mean()) * 100
    # break-even (assumptions, editable)
    FIXED = 600000   # ₹ / month for one metro café-bar: rent ~2.5L + staff ~2.2L + utilities & marketing ~1.3L
    MARGIN = .60     # gross margin after kava root import, ingredients, packaging
    med_spend_b = float(np.median(spend_raw.loc[buyers.index]))
    mean_spend_b = float(np.mean(np.clip(spend_raw.loc[buyers.index], 0, cap)))
    need_customers = FIXED / (med_spend_b * MARGIN)
    drinks_day = FIXED / (best_price * MARGIN) / 30
    save("breakeven", [
        {"item": "fixed_cost", "label": "Fixed cost per outlet per month (assumed)", "value": FIXED, "kind": "assumption"},
        {"item": "margin", "label": "Gross margin per drink (assumed)", "value": MARGIN, "kind": "assumption"},
        {"item": "price", "label": "Price per drink (revenue-best point from Q24)", "value": best_price, "kind": "from survey"},
        {"item": "median_spend", "label": "Median monthly spend of a buyer (Q25)", "value": round(med_spend_b), "kind": "from survey"},
        {"item": "customers_needed", "label": "Regular customers needed to break even", "value": round(need_customers), "kind": "result"},
        {"item": "drinks_day", "label": "Drinks a day needed to break even", "value": round(drinks_day), "kind": "result"},
    ])

    top_tier = next(t for t in tier_rows if t["tier"] == "Most probable")
    clf_best_row = next(x for x in clf_rows if x["model"] == best_clf)
    reg_best_row = next(x for x in reg_rows if x["best"] == 1)
    safety_share = next((b["all"] for b in counts("main_barrier") if b["answer"] == "Safety doubts"), 0)
    kpis = [
        {"metric": "respondents", "label": "Clean respondents", "value": len(df)},
        {"metric": "buy_rate", "label": "Say they would buy (Q30)", "value": r(df.would_buy.mean(), 4)},
        {"metric": "top_tier_share", "label": "Most-probable segment size", "value": top_tier["share"]},
        {"metric": "top_tier_buy", "label": "Buy rate in most-probable segment", "value": top_tier["buy_rate"]},
        {"metric": "top_tier_value_share", "label": "Share of expected revenue from most-probable segment", "value": top_tier["value_share"]},
        {"metric": "median_spend_buyers", "label": "Median monthly spend of a buyer (₹)", "value": round(med_spend_b)},
        {"metric": "mean_spend_buyers", "label": "Average monthly spend of a buyer, capped (₹)", "value": round(mean_spend_b)},
        {"metric": "median_max_price", "label": "Median max price per drink (₹)", "value": r(df.max_price_drink.median(), 0)},
        {"metric": "best_price", "label": "Revenue-best price per drink (₹)", "value": best_price},
        {"metric": "nps_all", "label": "NPS, all respondents", "value": r(nps(df.nps_score), 1)},
        {"metric": "nps_buyers", "label": "NPS, buyers", "value": r(nps(buyers.nps_score), 1)},
        {"metric": "clf_auc", "label": f"Classification AUC ({best_clf}, 5-fold)", "value": clf_best_row["cv_auc"]},
        {"metric": "clf_acc", "label": "Classification accuracy (test)", "value": clf_best_row["accuracy"]},
        {"metric": "reg_r2", "label": f"Regression R² on log spend ({best_reg})", "value": reg_best_row["r2"]},
        {"metric": "reg_mae", "label": "Regression MAE (₹)", "value": reg_best_row["mae"]},
        {"metric": "reg_medae", "label": "Regression median error (₹)", "value": reg_best_row["medae"]},
        {"metric": "reg_n", "label": "Buyers used for spend model", "value": len(bi)},
        {"metric": "k", "label": "Segments (k-means)", "value": K},
        {"metric": "rules", "label": "Association rules found", "value": arm_summary["rules_found"]},
        {"metric": "pair_rules", "label": "Rules of the form 'two services → a third'", "value": arm_summary["pair_rules"]},
        {"metric": "arm_buyers", "label": "Buyers used for rule mining", "value": nB},
        {"metric": "safety_barrier", "label": "Say safety doubts would stop them", "value": safety_share},
        {"metric": "customers_needed", "label": "Regular customers needed per outlet", "value": round(need_customers)},
        {"metric": "drinks_day", "label": "Drinks a day to break even", "value": round(drinks_day)},
        {"metric": "raw_n", "label": "Raw responses", "value": n0},
    ]
    save("kpis", kpis)

    # scorecard (rule-of-thumb thresholds, stated on the page)
    score = [
        {"check": "Enough people want it", "measure": "Say they would buy (Q30)", "value": r(df.would_buy.mean(), 4), "threshold": .30, "fmt": "pct", "rule": "≥ 30%"},
        {"check": "A clear core segment exists", "measure": "Most-probable segment share", "value": top_tier["share"], "threshold": .15, "fmt": "pct", "rule": "≥ 15% of market"},
        {"check": "Buyers can be targeted", "measure": "Classification AUC", "value": clf_best_row["cv_auc"], "threshold": .75, "fmt": "num", "rule": "≥ 0.75"},
        {"check": "Price covers a café-bar", "measure": "Revenue-best price (₹)", "value": best_price, "threshold": 200, "fmt": "inr", "rule": "≥ ₹200"},
        {"check": "Buyers would spread the word", "measure": "NPS among buyers", "value": r(nps(buyers.nps_score), 1), "threshold": 0, "fmt": "num", "rule": "> 0"},
        {"check": "Safety fear is manageable", "measure": "Safety doubts as main barrier", "value": safety_share, "threshold": .25, "fmt": "pct", "rule": "≤ 25%", "lower_is_better": 1},
    ]
    for s in score:
        s["pass"] = int(s["value"] <= s["threshold"]) if s.get("lower_is_better") else int(s["value"] > s["threshold"] if s["threshold"] == 0 else s["value"] >= s["threshold"])
    for s in score:
        s["status"] = "Pass" if s["pass"] else "Fail"
    save("scorecard", score)

    save("survey", [{"q": f"Q{q[0]}", "qnum": q[0], "section": q[1], "question": q[2], "type": q[3], "options": q[4],
                     "column": q[5], "feeds": q[6]} for q in QUESTIONS])

    # =============== 8. EVALUATION EXTRAS (baselines, gains, threshold, stability) ===============
    from sklearn.metrics import adjusted_rand_score
    yb_all = df.would_buy.values
    br = yb_all.mean()
    out["baselines"] = {
        "majority_acc": r(1 - br, 4), "buy_rate": r(br, 4),
        "buy_rate_ci": r(1.96 * np.sqrt(br * (1 - br) / len(yb_all)), 4),
    }
    gd = df[["p_buy", "would_buy"]].sort_values("p_buy", ascending=False).reset_index(drop=True)
    gd["decile"] = np.arange(len(gd)) * 10 // len(gd) + 1
    gains = []
    tot = gd.would_buy.sum()
    cum = 0
    for dcl, g in gd.groupby("decile"):
        cum += g.would_buy.sum()
        gains.append({"decile": int(dcl), "people_share": r(dcl / 10, 2), "buy_rate": r(g.would_buy.mean(), 4),
                      "buyers_captured": r(cum / tot, 4), "lift": r(g.would_buy.mean() / br, 3)})
    save("gains", gains)
    pt = bm.predict_proba(Xte)[:, 1]
    th = []
    for t in np.round(np.arange(0.2, 0.81, 0.05), 2):
        pr = (pt >= t).astype(int)
        th.append({"threshold": float(t), "precision": r(precision_score(yte, pr, zero_division=0), 4),
                   "recall": r(recall_score(yte, pr), 4), "flagged": r(pr.mean(), 4), "accuracy": r(accuracy_score(yte, pr), 4)})
    save("threshold", th)
    rng_s = np.random.default_rng(SEED)
    aris = []
    for b in range(8):
        sub = rng_s.choice(len(Z), int(0.8 * len(Z)), replace=False)
        kb = KMeans(K, n_init=10, random_state=100 + b).fit(Z[sub])
        aris.append(adjusted_rand_score(km.labels_, kb.predict(Z)))
    out["clu_stability"] = {"ari_mean": r(np.mean(aris), 3), "ari_min": r(np.min(aris), 3), "runs": len(aris)}
    zdf = pd.DataFrame(Z, columns=CLU, index=df.index)
    zdf["segment"] = df.cluster.map(segcode)
    zp = zdf.groupby("segment").mean()
    labels_q = {c: (f"Q{QLABEL[c][0]} " if c in QLABEL else "") + c.replace("_", " ") for c in CLU}
    save("clu_zprofile", [{"segment": sgm, "feature": labels_q[f], "z": r(zp.loc[sgm, f], 3)} for sgm in zp.index for f in CLU])
    med = np.median(yb[idx_tr])
    base_pred = np.full(len(y_eval), med)
    out["reg_baseline"] = {"r2": r(r2_score(np.log1p(y_eval), np.log1p(base_pred))),
                           "mae": r(mean_absolute_error(y_eval, base_pred), 0), "medae": r(np.median(np.abs(y_eval - base_pred)), 0)}
    out["clean"] = df
    out["segment_profiles"] = prof
    return out
