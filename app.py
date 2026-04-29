"""Streamlit dashboard for the Care Gap Outreach Engine."""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from care_gaps import detect_gaps
from prioritization import DEFAULT_WEIGHTS, score_gaps
from synthetic_data import generate_panel

st.set_page_config(
    page_title="Care Gap Outreach Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Generating synthetic panel...")
def _build_panel(n, seed):
    panel = generate_panel(n=n, seed=seed)
    gaps = detect_gaps(panel)
    return panel, gaps


def _score(panel, gaps, weights):
    return score_gaps(panel, gaps, weights=weights)


# --- Sidebar ---------------------------------------------------------------

with st.sidebar:
    st.title("Care Gap Engine")
    st.caption("Population health outreach prioritization, demo build")

    panel_size = st.slider("Patient panel size", 100, 2000, 1000, step=100)
    seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    st.subheader("Score weights")
    st.caption("Tune how the queue is ranked. Weights normalize automatically.")

    w_clin = st.slider("Clinical urgency", 0.0, 1.0, DEFAULT_WEIGHTS["clinical"], 0.05)
    w_resp = st.slider("Response likelihood", 0.0, 1.0, DEFAULT_WEIGHTS["response"], 0.05)
    w_eq = st.slider("Equity priority", 0.0, 1.0, DEFAULT_WEIGHTS["equity"], 0.05)

    total = w_clin + w_resp + w_eq
    if total == 0:
        total = 1
    weights = {
        "clinical": w_clin / total,
        "response": w_resp / total,
        "equity": w_eq / total,
    }

    st.divider()
    st.subheader("Outreach drafting")
    if os.getenv("ANTHROPIC_API_KEY"):
        st.success("Claude API key detected")
    else:
        st.warning("No ANTHROPIC_API_KEY set. Outreach drafting disabled.")
    n_to_draft = st.slider("How many top patients to draft for", 1, 20, 5)


# --- Data ------------------------------------------------------------------

panel, gaps = _build_panel(panel_size, int(seed))
scored = _score(panel, gaps, weights)

# --- Main ------------------------------------------------------------------

st.title("Care Gap Outreach Engine")
st.caption(
    "Prioritization first. Drafted messages second. Built for population health "
    "teams with HEDIS / value-based care contracts."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients in panel", f"{len(panel):,}")
c2.metric("Open care gaps", f"{len(gaps):,}")
c3.metric(
    "High-urgency gaps",
    f"{(scored['urgency'] == 'high').sum():,}",
)
c4.metric(
    "Patients with ≥1 gap",
    f"{scored['patient_id'].nunique():,}",
)

tab_queue, tab_drafts, tab_equity, tab_method = st.tabs(
    ["Outreach Queue", "Drafted Messages", "Equity Breakdown", "Methodology"]
)

# --- Tab: queue ------------------------------------------------------------

with tab_queue:
    st.subheader("Top of the outreach queue")
    st.caption("Sorted by priority score. Filters in the dropdowns below.")

    f1, f2, f3 = st.columns(3)
    measure_filter = f1.multiselect(
        "Measure",
        sorted(scored["measure"].unique()),
        default=sorted(scored["measure"].unique()),
    )
    lang_filter = f2.multiselect(
        "Language",
        sorted(scored["language"].unique()),
        default=sorted(scored["language"].unique()),
    )
    urg_filter = f3.multiselect(
        "Urgency", ["high", "medium", "low"], default=["high", "medium", "low"]
    )

    filtered = scored[
        scored["measure"].isin(measure_filter)
        & scored["language"].isin(lang_filter)
        & scored["urgency"].isin(urg_filter)
    ]

    st.dataframe(
        filtered[
            [
                "patient_id", "measure", "urgency", "age", "gender",
                "language", "insurance", "race_ethnicity", "barriers",
                "no_show_rate",
                "clinical_score", "response_score", "equity_score",
                "priority_score",
            ]
        ].head(50),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(f"Showing top 50 of {len(filtered):,} matching gaps.")

# --- Tab: drafted messages -------------------------------------------------

with tab_drafts:
    st.subheader("Personalized outreach drafts")

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error(
            "Set ANTHROPIC_API_KEY in a .env file to enable Claude-drafted messages. "
            "See .env.example."
        )
    else:
        if st.button(f"Draft messages for top {n_to_draft} patients", type="primary"):
            from outreach import generate_messages_batch

            with st.spinner("Calling Claude... cached system prompt means cost stays flat after the first call."):
                drafted, usage = generate_messages_batch(
                    panel, scored, top_n=n_to_draft
                )

            uc1, uc2, uc3, uc4 = st.columns(4)
            uc1.metric("Messages drafted", usage["n_messages"])
            uc2.metric("Total input tokens", f"{usage['total_input_tokens']:,}")
            uc3.metric("Total output tokens", f"{usage['total_output_tokens']:,}")
            uc4.metric(
                "Cache hits (input tokens)",
                f"{usage['total_cache_read_tokens']:,}",
                help="Tokens served from cache at ~10% of normal cost. The system prompt "
                     "is cached on the first call, so subsequent drafts are nearly free.",
            )

            for _, row in drafted.iterrows():
                with st.expander(
                    f"{row['patient_id']} · {row['measure']} · score {row['priority_score']} · "
                    f"{row['language']} · {row['health_literacy']} literacy"
                ):
                    cl, cr = st.columns([1, 1])
                    with cl:
                        st.markdown("**Patient context**")
                        st.write({
                            "age": int(row["age"]),
                            "gender": row["gender"],
                            "race": row["race_ethnicity"],
                            "language": row["language"],
                            "insurance": row["insurance"],
                            "barriers": row["barriers"],
                            "no-show rate": f"{row['no_show_rate']:.0%}",
                            "ZIP": row["zip_code"],
                            "literacy": row["health_literacy"],
                        })
                        st.markdown("**Score breakdown**")
                        st.write({
                            "clinical": row["clinical_score"],
                            "response": row["response_score"],
                            "equity": row["equity_score"],
                            "final": row["priority_score"],
                        })
                    with cr:
                        st.markdown("**Drafted message**")
                        st.info(row["drafted_message"])

# --- Tab: equity -----------------------------------------------------------

with tab_equity:
    st.subheader("Equity breakdown")
    st.caption(
        "If your outreach mix doesn't match your panel mix, you'll widen disparities, "
        "not close them. These charts compare gap rates and queue position across groups."
    )

    # Gap rate per language
    gap_per_patient = scored.groupby("patient_id").size().rename("n_gaps")
    by_lang = panel.merge(gap_per_patient, on="patient_id", how="left").fillna(0)

    col_l, col_r = st.columns(2)

    with col_l:
        rate = (
            by_lang.groupby("language")
            .agg(patients=("patient_id", "count"), gaps=("n_gaps", "sum"))
            .reset_index()
        )
        rate["gaps_per_patient"] = (rate["gaps"] / rate["patients"]).round(2)
        fig = px.bar(
            rate, x="language", y="gaps_per_patient",
            title="Open gaps per patient, by language",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        rate2 = (
            by_lang.groupby("insurance")
            .agg(patients=("patient_id", "count"), gaps=("n_gaps", "sum"))
            .reset_index()
        )
        rate2["gaps_per_patient"] = (rate2["gaps"] / rate2["patients"]).round(2)
        fig2 = px.bar(
            rate2, x="insurance", y="gaps_per_patient",
            title="Open gaps per patient, by insurance",
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # Queue position - what fraction of the top-100 gaps are from each subgroup
    top100 = scored.head(100)
    queue_lang = top100["language"].value_counts(normalize=True).rename("share").reset_index()
    queue_lang.columns = ["language", "share_of_top_100"]

    panel_lang = panel["language"].value_counts(normalize=True).rename("share").reset_index()
    panel_lang.columns = ["language", "share_of_panel"]

    compare = queue_lang.merge(panel_lang, on="language", how="outer").fillna(0)
    compare = compare.melt(id_vars="language", var_name="metric", value_name="share")

    fig3 = px.bar(
        compare, x="language", y="share", color="metric", barmode="group",
        title="Top 100 of queue vs full panel - language mix",
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

    st.caption(
        "If 'share of top 100' < 'share of panel' for a non-English-speaking subgroup, "
        "raise the equity weight in the sidebar. Watch how the queue rebalances."
    )

# --- Tab: methodology ------------------------------------------------------

with tab_method:
    st.subheader("How the prioritization works")
    st.markdown("""
**Priority score** = w_clinical × clinical_urgency
   + w_response × response_likelihood
   + w_equity × equity_priority

All three components are normalized to [0, 1] so the weights map directly to relative importance.

### Clinical urgency
- Each measure has a base weight reflecting how consequential it is when overdue
  (A1c for a diabetic > flu vaccine).
- Multiplied by an urgency tier (`high` / `medium` / `low`) based on how far past
  due the screening is, with diabetic A1c bumped to `high` regardless of duration.

### Response likelihood
- `1 - no_show_rate`. Patients who consistently show up convert outreach into
  closed gaps; high-no-show patients need a different intervention (community
  health worker, in-person visit, etc.) rather than another text message.
- This isn't about deprioritizing high-no-show patients - it's about routing
  them to the right channel.

### Equity priority
- Boosts patients facing structural access barriers: Medicaid / Uninsured,
  non-English language, Hispanic / Black race, high-need ZIPs.
- Each factor adds a fixed bump; the score is clipped to [0, 1].
- This is the lever pop health teams pull to actively close disparities.
- For NYU Langone Population Health specifically, equity-adjusted CMS measures
  are real revenue (HEDIS Health Equity Index, HHS / CMS rules effective 2024+).

### Outreach drafting
- Claude (`claude-opus-4-7`) drafts a personalized message per patient.
- Frozen system prompt is cached, so token cost stays flat as you scale to
  hundreds of drafts. Patient-specific context goes in the user turn.
- Adaptive thinking - simple cases get a quick reply; multi-barrier non-English
  cases get more reasoning.

### Data sources for the synthetic panel
- CDC adult prevalence rates (diabetes, hypertension, smoking, depression)
- BRFSS 2022 screening up-to-date rates
- Census ACS 2022 demographic distributions
- KFF 2022 insurance coverage breakdown
- Health Affairs / AAFP no-show rate priors

The panel is reproducible per `seed`. Not for clinical use - this is a portfolio demo.
""")
