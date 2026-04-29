"""Detect open care gaps per patient based on USPSTF / HEDIS rules.

Rules implemented (simplified - real systems handle many more edge cases):
- Mammography: women 40-74, every 24 months (USPSTF Grade B for 40-74)
- Colorectal cancer screening: 45-75, colonoscopy every 10 years
- Cervical cancer (Pap): women 21-65, every 36 months
- A1c: diabetics, every 6 months (ADA)
- BP check: hypertensives, every 12 months
- Flu vaccine: all adults, every 12 months (HEDIS)
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class Gap:
    patient_id: str
    measure: str
    overdue_days: int           # days past the due date
    urgency: str                # "high" / "medium" / "low"
    reason: str                 # short explanation for the cover-letter / outreach

    def to_dict(self):
        return {
            "patient_id": self.patient_id,
            "measure": self.measure,
            "overdue_days": self.overdue_days,
            "urgency": self.urgency,
            "reason": self.reason,
        }


# Days between recommended screenings.
INTERVAL = {
    "mammogram": 24 * 30,
    "colonoscopy": 120 * 30,        # ~10 yr
    "pap": 36 * 30,
    "a1c": 6 * 30,
    "bp_check": 12 * 30,
    "flu_vaccine": 12 * 30,
}


def _check(last_days, interval, *, never_eligible=False):
    """Return overdue-days if patient is overdue, 0 if up to date, -1 if N/A."""
    if never_eligible:
        return -1
    if last_days is None:
        # Eligible but no record - treat as max-overdue
        return interval * 2
    overdue = last_days - interval
    return overdue if overdue > 0 else 0


def _urgency(measure, overdue_days, has_diabetes=False):
    """Bucket overdue-days into clinical urgency. Diabetics with overdue A1C
    get bumped to high regardless of how long because the consequence is sharper."""
    if measure == "a1c" and has_diabetes and overdue_days > 0:
        return "high"
    if measure in {"colonoscopy", "mammogram"}:
        if overdue_days > 365:
            return "high"
        if overdue_days > 180:
            return "medium"
        return "low"
    if measure == "bp_check" and overdue_days > 365:
        return "high"
    if overdue_days > 365:
        return "medium"
    return "low"


def detect_gaps(panel: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (patient, gap)."""
    gaps = []
    for _, p in panel.iterrows():
        pid = p["patient_id"]

        # Mammography
        eligible = p["gender"] == "F" and 40 <= p["age"] <= 74
        d = _check(p["last_mammogram_days"], INTERVAL["mammogram"], never_eligible=not eligible)
        if d > 0:
            gaps.append(Gap(pid, "mammogram", d,
                            _urgency("mammogram", d),
                            f"Mammography overdue by {d // 30} months. Recommended every 2 years."))

        # Colorectal cancer screening
        eligible = 45 <= p["age"] <= 75
        d = _check(p["last_colonoscopy_days"], INTERVAL["colonoscopy"], never_eligible=not eligible)
        if d > 0:
            gaps.append(Gap(pid, "colonoscopy", d,
                            _urgency("colonoscopy", d),
                            f"Colonoscopy overdue by {d // 365} years. Recommended every 10 years."))

        # Cervical cancer screening
        eligible = p["gender"] == "F" and 21 <= p["age"] <= 65
        d = _check(p["last_pap_days"], INTERVAL["pap"], never_eligible=not eligible)
        if d > 0:
            gaps.append(Gap(pid, "pap", d,
                            _urgency("pap", d),
                            f"Pap test overdue by {d // 30} months. Recommended every 3 years."))

        # A1c for diabetics
        if p["has_diabetes"]:
            d = _check(p["last_a1c_days"], INTERVAL["a1c"])
            if d > 0:
                gaps.append(Gap(pid, "a1c", d,
                                _urgency("a1c", d, has_diabetes=True),
                                f"A1c overdue by {d // 30} months. Diabetics need testing every 6 months."))

        # BP check for hypertensives
        if p["has_hypertension"]:
            d = _check(p["last_bp_check_days"], INTERVAL["bp_check"])
            if d > 0:
                gaps.append(Gap(pid, "bp_check", d,
                                _urgency("bp_check", d),
                                f"Blood pressure check overdue by {d // 30} months."))

        # Flu vaccine - everyone
        d = _check(p["last_flu_vaccine_days"], INTERVAL["flu_vaccine"])
        if d > 0:
            gaps.append(Gap(pid, "flu_vaccine", d,
                            _urgency("flu_vaccine", d),
                            f"Flu vaccine overdue by {d // 30} months."))

    return pd.DataFrame([g.to_dict() for g in gaps])


if __name__ == "__main__":
    from synthetic_data import generate_panel
    panel = generate_panel(200)
    gaps = detect_gaps(panel)
    print(gaps.head(10))
    print(f"\n{len(gaps)} total gaps across {panel.shape[0]} patients")
    print(f"\ngaps by measure:\n{gaps['measure'].value_counts()}")
    print(f"\nby urgency:\n{gaps['urgency'].value_counts()}")
