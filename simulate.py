"""Simulate survey responses for My Meltdown.
Respondents are driven by 4 hidden traits (openness, health focus, social
nightlife, daily-ritual habit). Answers and outcomes are generated from these
traits plus noise, then dirty data is injected:
  - missing answers, straight-liners, typo outliers (extra zero),
    extreme spenders, impossible values, and label noise on Q30.
Usage: python simulate.py OUT.csv [N] [SEED]   or   from simulate import generate
"""
import sys
import numpy as np
import pandas as pd

def generate(N=3500, SEED=42):
    """Return a DataFrame of N simulated survey responses (with noise and outliers)."""
    rng = np.random.default_rng(SEED)

    def sig(x):
        return 1 / (1 + np.exp(-x))

    def likert(x):
        return np.clip(np.round(3 + x), 1, 5).astype(int)

    def pick(opts, p, n=None):
        return rng.choice(opts, size=n or N, p=p)

    # ---------- profile ----------
    AGES = ["18-20", "21-24", "25-29", "30-35", "36-45"]
    age = pick(AGES, [.17, .30, .25, .16, .12])
    age_i = pd.Series(age).map({a: i for i, a in enumerate(AGES)}).values
    gender = pick(["Male", "Female", "Other/Prefer not"], [.52, .45, .03])
    TIERS = ["Metro", "Tier 1", "Tier 2", "Tier 3"]
    tier = pick(TIERS, [.38, .30, .22, .10])
    tier_i = pd.Series(tier).map({t: i for i, t in enumerate(TIERS)}).values

    occ = np.empty(N, dtype=object)
    for i in range(N):
        if age_i[i] == 0:
            p = [.85, .04, .02, .07, .02]
        elif age_i[i] == 1:
            p = [.45, .33, .07, .11, .04]
        else:
            p = [.04, .58, .20, .13, .05]
        occ[i] = rng.choice(["Student", "Salaried", "Business owner", "Freelancer/Creator", "Not working"], p=p)

    base_inc = pd.Series(occ).map({"Student": 9000, "Salaried": 55000, "Business owner": 85000,
                                   "Freelancer/Creator": 40000, "Not working": 6000}).values
    tier_mult = np.array([1.35, 1.1, .85, .7])[tier_i]
    income = base_inc * tier_mult * rng.lognormal(0, .45, N) * (1 + .08 * np.maximum(age_i - 2, 0))
    income = np.round(income, -2)

    # ---------- hidden traits ----------
    openness = rng.normal(0, 1, N) + .25 * (2 - age_i) / 2 + .2 * (tier_i == 0)
    health = rng.normal(0, 1, N) + .15 * (gender == "Female")
    social = rng.normal(0, 1, N) + .35 * (age_i <= 2) - .3 * (age_i == 4) + .25 * (tier_i <= 1)
    ritual = rng.normal(0, 1, N) + .3 * (occ == "Salaried") + .2 * (tier_i == 0)
    inc_z = (np.log(income) - np.log(income).mean()) / np.log(income).std()

    # ---------- habits ----------
    alc_lat = .8 * social - .3 * health + rng.normal(0, .9, N) + .3 * (gender == "Male")
    alc_i = np.digitize(alc_lat, [-.65, .1, .75, 1.55])           # 0..4
    ALC = ["Never", "Rarely", "Monthly", "Weekly", "2+ per week"]
    alcohol = np.array(ALC)[alc_i]
    want_cut = likert(.55 * health + .35 * (alc_i >= 3) - .4 * (alc_i == 0) + rng.normal(0, .8, N))
    cafe_visits = rng.poisson(np.exp(1.5 + .35 * ritual + .2 * inc_z + .15 * (tier_i == 0)))
    bar_visits = rng.poisson(np.exp(.2 + .55 * social + .35 * (alc_i - 2) + .15 * inc_z).clip(0.05, 20))
    cafe_spend = np.round(np.exp(5.3 + .28 * inc_z + rng.normal(0, .3, N)), -1)
    nightout = np.round(np.exp(7.1 + .35 * inc_z + .15 * social + rng.normal(0, .35, N)), -1)
    wellness = likert(.9 * health + rng.normal(0, .6, N))
    stress = likert(.4 * (occ == "Salaried") + .3 * (occ == "Student") - .2 * health + rng.normal(0, .9, N))
    social_hrs = np.round(np.clip(rng.gamma(2.2, 1.3, N) + .5 * (age_i <= 1) + .4 * openness, .2, 14), 1)
    tried_na = (rng.random(N) < sig(-.6 + .6 * openness + .4 * health + .5 * (tier_i == 0))).astype(int)
    aw_lat = .7 * openness + .4 * (tier_i == 0) + rng.normal(0, 1, N)
    kava_aw = np.array(["Never heard", "Heard of it", "Tried it"])[np.digitize(aw_lat, [.75, 2.0])]

    # ---------- concept attitudes ----------
    buzz = likert(.55 * openness + .35 * social + .25 * (stress - 3) + .2 * (want_cut - 3) + rng.normal(0, .75, N))
    safety = likert(-.5 * openness + .3 * health + .3 * (age_i >= 3) + rng.normal(0, .8, N))
    novelty = likert(.95 * openness + rng.normal(0, .6, N))
    influ = likert(.45 * openness + .35 * (age_i <= 1) + .25 * (social_hrs - 3) / 2 + rng.normal(0, .8, N))

    t_lat = np.c_[ritual * 1.0 + .2, rng.normal(0, .7, N) + .4 * (occ == "Student"), social * .9, social * 1.1 - .2]
    TIMES = ["Morning", "Afternoon", "Evening", "Late night"]
    pref_time = np.array(TIMES)[np.argmax(t_lat + rng.gumbel(0, 1, (N, 4)), 1)]
    f_lat = np.c_[.4 * health, .6 * ritual, .3 * (occ == "Student") + .2 * health, .8 * social, .3 * inc_z + .2 * (tier_i == 0)]
    FORMATS = ["Hot brew", "Coffee blend", "Fruit cooler", "Kava cocktail", "Canned drink"]
    pref_fmt = np.array(FORMATS)[np.argmax(f_lat + rng.gumbel(0, 1, (N, 5)), 1)]

    # ---------- purchase intent (Q30) ----------
    alc_eff = np.array([-.35, .1, .4, .45, .05])[alc_i]
    age_eff = np.array([.2, .5, .4, 0, -.55])[age_i]
    tier_eff = np.array([.45, .2, -.25, -.6])[tier_i]
    aw_eff = pd.Series(kava_aw).map({"Never heard": 0, "Heard of it": .2, "Tried it": .6}).values
    z = (-1.75 + age_eff + tier_eff + alc_eff + .3 * (want_cut - 3) + .2 * (wellness - 3) + .12 * (stress - 3)
         + .6 * (buzz - 3) - .45 * (safety - 3) + .35 * (novelty - 3) + .18 * (influ - 3)
         + .4 * tried_na + aw_eff + .03 * np.minimum(cafe_visits, 20) + .1 * inc_z + rng.normal(0, .8, N))
    p_buy = sig(z)
    would_buy = (rng.random(N) < p_buy).astype(int)

    # ---------- services (Q23, multi-select) ----------
    night = social + .3 * (buzz - 3)
    daily = ritual + .2 * (pref_time == "Morning")
    well = health + .3 * (stress - 3)
    conv = .5 * inc_z + .3 * (tier_i == 0) + rng.normal(0, .6, N)
    buy_shift = np.where(would_buy == 1, 0.0, -1.6)
    svc_lat = {
        "S1": -.4 + 1.1 * daily,
        "S2": -.8 + .5 * (occ == "Student") + .4 * (pref_time == "Afternoon") + .3 * openness,
        "S3": -.3 + 1.2 * night,
        "S4": -.9 + .6 * conv + .3 * (pref_fmt == "Canned drink"),
        "S5": -.5 + .7 * night + .3 * daily,
        "S6": -1.4 + .8 * inc_z + .4 * well,
        "S7": -1.3 + 1.1 * well,
        "S8": -.8 + 1.1 * night + .3 * (age_i <= 1),
        "S9": -1.6 + .9 * daily + .4 * inc_z,
        "S10": -1.0 + .9 * conv + .3 * daily,
    }
    svc = {k: (rng.random(N) < sig(v + buy_shift + rng.normal(0, .4, N))).astype(int) for k, v in svc_lat.items()}
    # coupling: membership users lean on delivery; party-goers try partner venues
    svc["S10"] = np.maximum(svc["S10"], ((svc["S9"] == 1) & (svc["S1"] == 1) & (rng.random(N) < .55)).astype(int))
    svc["S5"] = np.maximum(svc["S5"], ((svc["S3"] == 1) & (svc["S8"] == 1) & (rng.random(N) < .5)).astype(int))
    svc["S6"] = np.maximum(svc["S6"], ((svc["S7"] == 1) & (inc_z > 0) & (rng.random(N) < .45)).astype(int))

    # ---------- spend & intent (Q24-Q29) ----------
    max_price = np.round(np.exp(5.25 + .25 * inc_z + .15 * (buzz - 3) + .1 * (tier_i == 0) + rng.normal(0, .25, N)), -1)
    habit = .35 * np.log1p(cafe_visits) + .25 * np.log1p(bar_visits) + .15 * (buzz - 3) + .5 * (p_buy - .5)
    visits_int = np.where(would_buy == 1,
                          np.maximum(1, rng.poisson(np.exp(.35 + habit))),
                          rng.binomial(1, .25, N))
    ticket = (.55 * cafe_spend + .45 * max_price) * rng.uniform(.8, 1.05, N)
    monthly_spend = np.round(visits_int * ticket * rng.lognormal(0, .3, N), -1)
    nps = np.clip(np.round(np.where(would_buy == 1, 7.2, 4.5) + .4 * (buzz - 3) - .2 * (safety - 3) + rng.normal(0, 1.7, N)), 0, 10).astype(int)
    bar_lat = np.c_[-.3 * inc_z + .2 * (occ == "Student"), .7 * (safety - 3), .2 + rng.normal(0, .4, N),
                    .3 * (tier_i >= 2) + .1, .3 * (age_i >= 3) - .2, 1.2 * (p_buy - .55)]
    BARRIERS = ["Price", "Safety doubts", "Taste", "Not available nearby", "What people think", "Nothing"]
    barrier = np.array(BARRIERS)[np.argmax(bar_lat + rng.gumbel(0, .9, (N, 6)), 1)]
    ch_lat = np.c_[.4 * night + .2 * (tier_i == 0), .5 * daily, .5 * inc_z + .2 * well - .3, .5 * conv]
    CHANNELS = ["Own Meltdown café-bar", "Partner restaurant/café", "Hotel/resort", "Delivery/retail"]
    channel = np.array(CHANNELS)[np.argmax(ch_lat + rng.gumbel(0, 1, (N, 4)), 1)]

    df = pd.DataFrame({
        "respondent_id": [f"R{i:05d}" for i in range(1, N + 1)],
        "age_group": age, "gender": gender, "city_tier": tier, "occupation": occ, "monthly_income": income,
        "alcohol_freq": alcohol, "want_cut_alcohol": want_cut, "cafe_visits_month": cafe_visits,
        "bar_visits_month": bar_visits, "cafe_spend_visit": cafe_spend, "nightout_spend": nightout,
        "wellness": wellness, "stress": stress, "social_media_hrs": social_hrs, "tried_na_alt": tried_na,
        "kava_awareness": kava_aw, "buzz_appeal": buzz, "safety_concern": safety, "novelty_seeking": novelty,
        "influencer_effect": influ, "preferred_time": pref_time, "preferred_format": pref_fmt,
        **svc,
        "max_price_drink": max_price, "monthly_spend": monthly_spend, "intended_visits": visits_int,
        "nps_score": nps, "main_barrier": barrier, "preferred_channel": channel, "would_buy": would_buy,
    })

    # ---------- dirty data ----------
    dirt = np.full(N, "", dtype=object)
    idx = rng.permutation(N)
    LIK = ["want_cut_alcohol", "wellness", "stress", "buzz_appeal", "safety_concern", "novelty_seeking", "influencer_effect"]
    # 1.5% straight-liners: same Likert answer everywhere, random intent
    sl = idx[:int(.015 * N)]
    for i in sl:
        v = rng.integers(1, 6)
        df.loc[i, LIK] = v
        df.loc[i, "would_buy"] = rng.integers(0, 2)
        dirt[i] = "straight-liner"
    # 1% income typo (extra zero) + a few zero incomes
    ty = idx[len(sl):len(sl) + int(.01 * N)]
    df.loc[ty, "monthly_income"] *= 10
    dirt[ty] = "income typo x10"
    zi = idx[len(sl) + len(ty):len(sl) + len(ty) + 15]
    df.loc[zi, "monthly_income"] = 0
    dirt[zi] = "zero income"
    # 1.5% extreme monthly spend among buyers
    buyers = np.where(df.would_buy.values == 1)[0]
    ex = rng.choice(buyers, int(.015 * N), replace=False)
    df.loc[ex, "monthly_spend"] = np.round(rng.uniform(25000, 120000, len(ex)), -2)
    dirt[ex] = np.where(dirt[ex] == "", "extreme spender", dirt[ex])
    # 1% extreme café spend, 0.5% impossible social media hours, a few huge café visit counts
    ec = rng.choice(N, int(.01 * N), replace=False)
    df.loc[ec, "cafe_spend_visit"] = np.round(rng.uniform(3000, 15000, len(ec)), -2)
    im = rng.choice(N, int(.005 * N), replace=False)
    df.loc[im, "social_media_hrs"] = rng.choice([20, 24, 30, 48], len(im))
    cv = rng.choice(N, 12, replace=False)
    df.loc[cv, "cafe_visits_month"] = rng.integers(60, 200, len(cv))
    # 3% label noise on Q30
    ln = rng.choice(N, int(.03 * N), replace=False)
    df.loc[ln, "would_buy"] = 1 - df.loc[ln, "would_buy"]
    # missing answers
    for col, rate in [("monthly_income", .04), ("cafe_spend_visit", .03), ("nightout_spend", .035),
                      ("social_media_hrs", .02), ("max_price_drink", .025), ("kava_awareness", .015)]:
        m = rng.random(N) < rate
        df.loc[m, col] = np.nan

    df["injected_issue"] = dirt  # kept for checking; a real survey would not have this column
    return df


if __name__ == "__main__":
    OUT = sys.argv[1] if len(sys.argv) > 1 else "my_meltdown_survey.csv"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 3500
    SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    df = generate(N, SEED)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df)} rows, buy rate {df.would_buy.mean():.3f}")
