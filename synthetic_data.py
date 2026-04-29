"""Generate a synthetic primary care patient panel.

Prevalence rates and demographic mixes are based on CDC, BRFSS, and ACS public
data so the panel is reasonable to play with. Not for clinical use - this is
for a portfolio demo.
"""

import random
from datetime import date

import numpy as np
import pandas as pd

# Adult prevalence baselines (CDC, NHIS, BRFSS).
DIABETES_PREV = 0.116
HTN_PREV = 0.477
HYPERLIPIDEMIA_PREV = 0.38
OBESITY_PREV = 0.419
SMOKER_PREV = 0.115
DEPRESSION_PREV = 0.082

# BRFSS - share of eligible adults UP TO DATE on each screen.
# We invert these to get overdue rates.
MAMMO_UP_TO_DATE = 0.755
COLO_UP_TO_DATE = 0.715
PAP_UP_TO_DATE = 0.802
A1C_UP_TO_DATE = 0.78
BP_RECENT = 0.92
FLU_UP_TO_DATE = 0.49

RACE_DIST = {
    "White (non-Hispanic)": 0.589,
    "Hispanic": 0.192,
    "Black": 0.134,
    "Asian": 0.062,
    "Other": 0.023,
}

# Slightly boosted Spanish + Mandarin so the demo shows multilingual outreach.
LANGUAGE_DIST = {
    "English": 0.72,
    "Spanish": 0.20,
    "Mandarin": 0.04,
    "Haitian Creole": 0.02,
    "Russian": 0.01,
    "Vietnamese": 0.01,
}

INSURANCE_DIST = {
    "Commercial": 0.50,
    "Medicare": 0.19,
    "Medicaid": 0.19,
    "Marketplace": 0.07,
    "Uninsured": 0.05,
}

# No-show priors by insurance type. Higher rates reflect access barriers,
# not patient fault. Numbers from primary care literature (Health Affairs, AAFP).
NO_SHOW_PRIOR = {
    "Commercial": 0.10,
    "Medicare": 0.09,
    "Medicaid": 0.22,
    "Marketplace": 0.13,
    "Uninsured": 0.27,
}

BARRIERS = [
    "transportation",
    "cost",
    "fear/anxiety",
    "language",
    "work schedule",
    "childcare",
    "low health literacy",
    "prior bad experience",
]
BARRIER_WEIGHTS = [0.20, 0.16, 0.11, 0.07, 0.21, 0.10, 0.08, 0.07]

# NYC-area ZIPs roughly tiered by community need (proxy for HRSA MUA designation).
HIGH_NEED_ZIPS = ["10453", "10456", "10468", "11206", "11212", "11221", "11233"]
MED_NEED_ZIPS = ["10025", "10031", "11216", "11226", "11378", "11432"]
LOW_NEED_ZIPS = ["10014", "10075", "10128", "11201", "11215", "11217", "11231"]


def _pick(options):
    return random.choices(list(options.keys()), weights=list(options.values()), k=1)[0]


def _diabetes_age_mult(age):
    # CDC: ~5% under 45, ~18% in 45-64, ~29% in 65+
    if age < 45:
        return 0.4
    if age < 65:
        return 1.5
    return 2.5


def _htn_age_mult(age):
    if age < 45:
        return 0.5
    if age < 65:
        return 1.2
    return 1.7


def _make_patient(pid):
    age = int(np.clip(np.random.normal(52, 18), 18, 92))
    gender = random.choices(["F", "M"], weights=[0.52, 0.48], k=1)[0]
    race = _pick(RACE_DIST)
    language = _pick(LANGUAGE_DIST)
    insurance = _pick(INSURANCE_DIST)

    has_dm = random.random() < DIABETES_PREV * _diabetes_age_mult(age)
    has_htn = random.random() < HTN_PREV * _htn_age_mult(age)
    has_hld = random.random() < HYPERLIPIDEMIA_PREV
    has_obese = random.random() < OBESITY_PREV
    smoker = random.random() < SMOKER_PREV
    has_dep = random.random() < DEPRESSION_PREV

    # Days since last completed screening. None = not eligible or never done.
    mammo_days = None
    if gender == "F" and 40 <= age <= 74:
        if random.random() < MAMMO_UP_TO_DATE:
            mammo_days = random.randint(30, 23 * 30)
        else:
            mammo_days = random.randint(24 * 30, 60 * 30)

    colo_days = None
    if 45 <= age <= 75:
        if random.random() < COLO_UP_TO_DATE:
            colo_days = random.randint(30, 119 * 30)
        else:
            colo_days = random.randint(121 * 30, 200 * 30)

    pap_days = None
    if gender == "F" and 21 <= age <= 65:
        if random.random() < PAP_UP_TO_DATE:
            pap_days = random.randint(30, 35 * 30)
        else:
            pap_days = random.randint(36 * 30, 84 * 30)

    a1c_days = None
    if has_dm:
        if random.random() < A1C_UP_TO_DATE:
            a1c_days = random.randint(30, 6 * 30)
        else:
            a1c_days = random.randint(6 * 30 + 1, 24 * 30)

    bp_days = None
    if has_htn:
        if random.random() < BP_RECENT:
            bp_days = random.randint(7, 12 * 30)
        else:
            bp_days = random.randint(13 * 30, 36 * 30)

    if random.random() < FLU_UP_TO_DATE:
        flu_days = random.randint(7, 11 * 30)
    else:
        flu_days = random.randint(12 * 30 + 1, 36 * 30)

    last_visit_days = random.randint(7, 540)

    # Insurance-anchored no-show with patient-level noise. Cap at 60%.
    base = NO_SHOW_PRIOR[insurance]
    no_show = float(np.clip(np.random.normal(base, 0.07), 0.0, 0.60))

    # ZIP - Medicaid/Uninsured pulled toward higher-need ZIPs
    if insurance in {"Medicaid", "Uninsured"}:
        zip_pool = HIGH_NEED_ZIPS
    else:
        zip_pool = random.choices(
            [HIGH_NEED_ZIPS, MED_NEED_ZIPS, LOW_NEED_ZIPS],
            weights=[0.2, 0.4, 0.4],
            k=1,
        )[0]
    zip_code = random.choice(zip_pool)

    n_barr = random.choices([0, 1, 2], weights=[0.30, 0.50, 0.20], k=1)[0]
    barriers = []
    if n_barr:
        barriers = list(set(random.choices(BARRIERS, weights=BARRIER_WEIGHTS, k=n_barr)))

    # National Assessment of Adult Literacy buckets
    literacy = random.choices(
        ["low", "standard", "high"], weights=[0.36, 0.50, 0.14], k=1
    )[0]

    return {
        "patient_id": f"P{pid:05d}",
        "age": age,
        "gender": gender,
        "race_ethnicity": race,
        "language": language,
        "insurance": insurance,
        "zip_code": zip_code,
        "last_visit_days": last_visit_days,
        "has_diabetes": has_dm,
        "has_hypertension": has_htn,
        "has_hyperlipidemia": has_hld,
        "has_obesity": has_obese,
        "is_smoker": smoker,
        "has_depression": has_dep,
        "last_mammogram_days": mammo_days,
        "last_colonoscopy_days": colo_days,
        "last_pap_days": pap_days,
        "last_a1c_days": a1c_days,
        "last_bp_check_days": bp_days,
        "last_flu_vaccine_days": flu_days,
        "no_show_rate": round(no_show, 3),
        "barriers": "; ".join(barriers) if barriers else "none",
        "health_literacy": literacy,
    }


def generate_panel(n=1000, seed=42):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    rows = [_make_patient(i) for i in range(1, n + 1)]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_panel(100)
    print(df.head())
    print(f"\n{len(df)} patients")
    print(f"diabetes: {df['has_diabetes'].mean():.1%}")
    print(f"htn: {df['has_hypertension'].mean():.1%}")
    print(f"median no-show: {df['no_show_rate'].median():.1%}")
