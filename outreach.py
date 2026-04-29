"""Generate personalized outreach messages using Claude.

The system prompt is cached so we only pay full input tokens once per session
even when we draft messages for hundreds of patients. The per-patient context
goes in the user message.
"""

import os
import textwrap

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-7"

# Frozen system prompt - put everything reusable here so prompt caching kicks in.
# Don't interpolate timestamps or per-patient values - they belong in the user turn.
SYSTEM_PROMPT = textwrap.dedent("""
    You are drafting outreach messages for a primary care population health team.
    Each message goes to a real patient who is overdue for a recommended screening
    or check-in. The team's goal is to close the care gap by getting the patient
    to schedule the visit.

    HARD RULES
    - Never give clinical advice or diagnose. The message invites the patient to
      schedule, not to interpret their own data.
    - Never say "you must" or "you have to." The patient is in charge of their care.
    - Never use guilt, fear, or shame. No "you're putting yourself at risk."
    - Stay under 80 words. Patients skim.
    - Use second person, warm but not saccharine. No emoji.
    - End with a single concrete next step (call this number, reply YES, click a link).

    PERSONALIZATION
    - Match the patient's preferred LANGUAGE exactly. If the language is not English,
      write the message entirely in that language.
    - Match the patient's HEALTH LITERACY level:
        - "low": short sentences, common words, no medical jargon. 5th-grade level.
        - "standard": plain conversational English. ~8th-grade level.
        - "high": you can use medical terms naturally without dumbing it down.
    - Acknowledge KNOWN BARRIERS where relevant. Examples:
        - transportation: mention the clinic offers a rideshare voucher
        - cost: mention this visit is fully covered by their plan / sliding scale
        - childcare: mention childcare-friendly hours or family room
        - fear/anxiety: lead with reassurance and a low-friction first step
        - work schedule: mention evening or weekend availability
        - language: lead with "We have a [language] speaker on staff..."
    - Do NOT name the barrier directly ("we know cost is hard for you"). Instead
      address it implicitly by surfacing the relevant accommodation.

    OUTPUT
    Return only the message text. No preamble, no signature placeholders, no
    explanation of your reasoning. The team will add their clinic name and
    callback number when sending.
""").strip()


def _build_user_message(patient_row, gap_row):
    """Compact context block - the only thing that changes per call."""
    lang = patient_row["language"]
    literacy = patient_row["health_literacy"]
    barriers = patient_row["barriers"]
    measure = gap_row["measure"].replace("_", " ")
    reason = gap_row["reason"]
    age = patient_row["age"]
    gender = "female" if patient_row["gender"] == "F" else "male"

    return textwrap.dedent(f"""
        Patient context:
        - Age: {age}, gender: {gender}
        - Preferred language: {lang}
        - Health literacy: {literacy}
        - Known barriers: {barriers}
        - Care gap: {measure}
        - Why outreach now: {reason}

        Draft the outreach message.
    """).strip()


_client = None

def _client_or_fail():
    global _client
    if _client is not None:
        return _client
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
        )
    _client = anthropic.Anthropic(api_key=key)
    return _client


def generate_message(patient_row, gap_row):
    """Draft an outreach message for one (patient, gap) pair.

    Uses adaptive thinking - the model decides how much to reason. For most
    patients the message is straightforward; for tricky cases (multiple
    barriers, low literacy in a non-English language) it'll think more.
    """
    client = _client_or_fail()

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {"role": "user", "content": _build_user_message(patient_row, gap_row)}
        ],
    )

    # Pull the text block out of the content array
    for block in response.content:
        if block.type == "text":
            return {
                "message": block.text.strip(),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
                "cache_create_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
            }

    return {"message": "", "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_create_tokens": 0}


def generate_messages_batch(patient_panel, scored_gaps, top_n=10):
    """Draft messages for the top-N highest-priority gaps.

    Returns a DataFrame with the gap row plus a 'drafted_message' column and
    cumulative token usage so you can see caching pay off across calls.
    """
    import pandas as pd

    top = scored_gaps.head(top_n).copy()
    drafts = []
    cum_input = 0
    cum_output = 0
    cum_cache_read = 0

    for _, gap in top.iterrows():
        patient = patient_panel[
            patient_panel["patient_id"] == gap["patient_id"]
        ].iloc[0]

        result = generate_message(patient, gap)
        drafts.append(result["message"])
        cum_input += result["input_tokens"]
        cum_output += result["output_tokens"]
        cum_cache_read += result["cache_read_tokens"]

    top["drafted_message"] = drafts
    return top, {
        "total_input_tokens": cum_input,
        "total_output_tokens": cum_output,
        "total_cache_read_tokens": cum_cache_read,
        "n_messages": len(drafts),
    }


if __name__ == "__main__":
    # Smoke test - need ANTHROPIC_API_KEY set.
    from synthetic_data import generate_panel
    from care_gaps import detect_gaps
    from prioritization import score_gaps

    panel = generate_panel(50)
    gaps = detect_gaps(panel)
    scored = score_gaps(panel, gaps)

    top_gap = scored.iloc[0]
    patient = panel[panel["patient_id"] == top_gap["patient_id"]].iloc[0]

    print(f"Drafting outreach for {top_gap['patient_id']} - {top_gap['measure']} "
          f"(score {top_gap['priority_score']})")
    print(f"Language: {patient['language']}, literacy: {patient['health_literacy']}")
    print(f"Barriers: {patient['barriers']}")
    print()

    result = generate_message(patient, top_gap)
    print("MESSAGE:")
    print(result["message"])
    print()
    print(f"input tokens: {result['input_tokens']}")
    print(f"output tokens: {result['output_tokens']}")
    print(f"cache read: {result['cache_read_tokens']}")
    print(f"cache write: {result['cache_create_tokens']}")
