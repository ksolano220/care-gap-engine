"""Score open care gaps so we can rank the outreach queue.

Final score = w_clin * clinical_urgency
            + w_resp * response_likelihood
            + w_eq   * equity_priority

All three components are normalized 0-1 so the weights map cleanly to relative
importance. Defaults skew toward clinical urgency because that's what closes
HEDIS measures, but you can re-weight in the dashboard.
"""

import pandas as pd

# Default weights. Tunable in the UI.
DEFAULT_WEIGHTS = {
    "clinical": 0.50,
    "response": 0.25,
    "equity": 0.25,
}

# Higher = more clinically consequential when overdue.
MEASURE_WEIGHT = {
    "a1c": 1.00,           # bad outcomes accumulate fast for uncontrolled diabetes
    "colonoscopy": 0.90,
    "mammogram": 0.85,
    "bp_check": 0.75,
    "pap": 0.65,
    "flu_vaccine": 0.45,
}

URGENCY_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3}


def _clinical_score(measure, urgency):
    base = MEASURE_WEIGHT.get(measure, 0.5)
    u = URGENCY_WEIGHT.get(urgency, 0.3)
    return min(base * u, 1.0)


def _response_score(no_show_rate):
    """How likely the patient is to actually respond / show up.
    Inverts no-show rate so 0% no-show = 1.0, 60% no-show = 0.4."""
    return max(0.0, 1.0 - no_show_rate)


def _equity_score(insurance, race_ethnicity, zip_code, language):
    """Boost patients facing structural barriers - this is where pop health
    teams put their thumb on the scale to close disparities. Each factor adds
    a small bump; clipped to [0, 1]."""
    s = 0.0
    if insurance in {"Medicaid", "Uninsured"}:
        s += 0.35
    if insurance == "Marketplace":
        s += 0.10
    if race_ethnicity in {"Hispanic", "Black"}:
        s += 0.20
    if language != "English":
        s += 0.15
    # ZIPs already filtered upstream - high-need ZIPs in the Bronx / Brooklyn
    if zip_code.startswith(("104", "112")):
        s += 0.15
    return min(s, 1.0)


def score_gaps(
    panel: pd.DataFrame,
    gaps: pd.DataFrame,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Return gaps DataFrame with score columns added, sorted high-to-low."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Join so each gap row has the patient context it needs
    df = gaps.merge(panel, on="patient_id", how="left")

    df["clinical_score"] = df.apply(
        lambda r: _clinical_score(r["measure"], r["urgency"]), axis=1
    )
    df["response_score"] = df["no_show_rate"].apply(_response_score)
    df["equity_score"] = df.apply(
        lambda r: _equity_score(
            r["insurance"], r["race_ethnicity"], r["zip_code"], r["language"]
        ),
        axis=1,
    )

    df["priority_score"] = (
        weights["clinical"] * df["clinical_score"]
        + weights["response"] * df["response_score"]
        + weights["equity"] * df["equity_score"]
    )

    df["priority_score"] = df["priority_score"].round(3)

    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from synthetic_data import generate_panel
    from care_gaps import detect_gaps

    panel = generate_panel(500)
    gaps = detect_gaps(panel)
    scored = score_gaps(panel, gaps)
    print(scored[[
        "patient_id", "measure", "urgency", "clinical_score",
        "response_score", "equity_score", "priority_score",
    ]].head(15))
    print(f"\ntop 15 of {len(scored)} gaps")
