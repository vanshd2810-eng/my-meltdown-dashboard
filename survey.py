"""My Meltdown - 30-question validation survey.
Each question is tagged with the model(s) it feeds:
  CLF = classification feature, CLF-T = classification target
  CLU = clustering feature, PROF = cluster profiling only
  ARM = association rule mining basket
  REG = regression feature, REG-T = regression target
  BIZ = business KPI (pricing, barriers, channel, loyalty)
"""

SERVICES = [
    ("S1", "Morning kava coffee / latte"),
    ("S2", "Afternoon kava fruit coolers"),
    ("S3", "Night kava cocktails (kavatails)"),
    ("S4", "Canned ready-to-drink kava to take home"),
    ("S5", "Meltdown menu at partner restaurants & cafés"),
    ("S6", "Kava lounge at hotels / resorts"),
    ("S7", "Wellness / yoga retreat kava sessions"),
    ("S8", "Pop-up events & parties"),
    ("S9", "Monthly membership / subscription"),
    ("S10", "Home delivery via quick-commerce apps"),
]

LIKERT = "1 = Strongly disagree … 5 = Strongly agree"

QUESTIONS = [
    # Section A - who you are
    (1, "A. Profile", "What is your age group?", "Single choice", "18–20 / 21–24 / 25–29 / 30–35 / 36–45", "age_group", "CLF, REG, PROF"),
    (2, "A. Profile", "What is your gender?", "Single choice", "Male / Female / Other or prefer not to say", "gender", "CLF, REG, PROF"),
    (3, "A. Profile", "Which type of city do you live in?", "Single choice", "Metro / Tier 1 / Tier 2 / Tier 3", "city_tier", "CLF, REG, PROF"),
    (4, "A. Profile", "What is your current occupation?", "Single choice", "Student / Salaried / Business owner / Freelancer or creator / Not working", "occupation", "CLF, REG, PROF"),
    (5, "A. Profile", "What is your monthly income or pocket money (₹)?", "Number", "₹ amount", "monthly_income", "CLF, CLU, REG"),
    # Section B - current habits
    (6, "B. Habits", "How often do you drink alcohol?", "Single choice", "Never / Rarely (<1 a month) / Monthly / Weekly / 2+ times a week", "alcohol_freq", "CLF, CLU, REG"),
    (7, "B. Habits", "I want to cut down on alcohol.", "Likert 1–5", LIKERT, "want_cut_alcohol", "CLF, CLU, REG"),
    (8, "B. Habits", "How many times a month do you visit a café?", "Number", "visits / month", "cafe_visits_month", "CLF, CLU, REG"),
    (9, "B. Habits", "How many times a month do you go to a bar, pub or club?", "Number", "visits / month", "bar_visits_month", "CLF, CLU, REG"),
    (10, "B. Habits", "How much do you usually spend per café visit (₹)?", "Number", "₹ per visit", "cafe_spend_visit", "CLF, CLU, REG"),
    (11, "B. Habits", "How much do you usually spend on a night out (₹)?", "Number", "₹ per night out", "nightout_spend", "CLF, CLU, REG"),
    (12, "B. Habits", "I actively take care of my health and wellness.", "Likert 1–5", LIKERT, "wellness", "CLF, CLU, REG"),
    (13, "B. Habits", "I feel stressed during a typical week.", "Likert 1–5", LIKERT, "stress", "CLF, CLU, REG"),
    (14, "B. Habits", "How many hours a day do you spend on social media?", "Number", "hours / day", "social_media_hrs", "CLF, CLU, REG"),
    (15, "B. Habits", "Have you tried non-alcoholic alternatives (mocktails, zero-proof spirits, NA beer)?", "Yes / No", "Yes / No", "tried_na_alt", "CLF, REG"),
    (16, "B. Habits", "Have you heard of or tried kava?", "Single choice", "Never heard / Heard of it / Tried it", "kava_awareness", "CLF, REG, PROF"),
    # Section C - attitude to the concept (shown after a 2-line concept description)
    (17, "C. Concept", "A drink that relaxes me like alcohol, without a hangover, appeals to me.", "Likert 1–5", LIKERT, "buzz_appeal", "CLF, CLU, REG"),
    (18, "C. Concept", "I would worry about the safety of a new plant-based relaxing drink.", "Likert 1–5", LIKERT, "safety_concern", "CLF, CLU, REG"),
    (19, "C. Concept", "I like trying new and unusual foods and drinks.", "Likert 1–5", LIKERT, "novelty_seeking", "CLF, CLU, REG"),
    (20, "C. Concept", "Influencers and trends affect what I eat and drink.", "Likert 1–5", LIKERT, "influencer_effect", "CLF, CLU, REG"),
    (21, "C. Concept", "When would you most likely have a kava drink?", "Single choice", "Morning / Afternoon / Evening / Late night", "preferred_time", "CLF, PROF"),
    (22, "C. Concept", "Which format would you most like?", "Single choice", "Hot brew / Coffee blend / Fruit cooler / Kava cocktail / Canned drink", "preferred_format", "CLF, PROF"),
    # Section D - services
    (23, "D. Services", "Which of these My Meltdown services would you use? (tick all that apply)", "Multi-select (10)",
     " / ".join(s for _, s in SERVICES), "S1 … S10", "ARM"),
    # Section E - spend and intent
    (24, "E. Spend & intent", "What is the most you would pay for one kava drink (₹)?", "Number", "₹ per drink", "max_price_drink", "BIZ (price curve)"),
    (25, "E. Spend & intent", "How much would you spend on My Meltdown in a month (₹)?", "Number", "₹ per month", "monthly_spend", "REG-T"),
    (26, "E. Spend & intent", "How many times a month would you visit or order?", "Number", "times / month", "intended_visits", "BIZ"),
    (27, "E. Spend & intent", "How likely are you to recommend My Meltdown to a friend? (0–10)", "Scale 0–10", "0 = Not at all … 10 = Extremely", "nps_score", "BIZ (NPS)"),
    (28, "E. Spend & intent", "What would most stop you from buying?", "Single choice", "Price / Safety doubts / Taste / Not available nearby / What people think / Nothing", "main_barrier", "BIZ"),
    (29, "E. Spend & intent", "Where would you most like to buy it?", "Single choice", "Own Meltdown café-bar / Partner restaurant or café / Hotel or resort / Delivery or retail", "preferred_channel", "BIZ"),
    (30, "E. Spend & intent", "Overall, would you buy My Meltdown drinks?", "Yes / No", "Yes / No", "would_buy", "CLF-T"),
]
