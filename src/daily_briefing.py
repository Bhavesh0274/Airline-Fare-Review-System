"""Phase 7: turn today's priority queue into a briefing a revenue-management
analyst can read in ten seconds, same pattern as the other two projects'
GenAI layer. Calls the Anthropic API when ANTHROPIC_API_KEY is set; otherwise
falls back to a deterministic template so the pipeline never breaks.

Model choice: Haiku, not Sonnet/Opus -- same reasoning as before. This is
bounded summarization over numbers that are already computed and correct;
there's no extra reasoning a bigger model would add.
"""
import json
import os
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL = "claude-haiku-4-5-20251001"
TOP_N = 8


def build_queue_summary() -> list[dict]:
    df = pd.read_csv(PROCESSED_DIR / "flights_final_policy.csv")
    top = df.sort_values("priority_score", ascending=False).head(TOP_N)
    return [
        {
            "route": r.route, "flight_date": r.flightDate, "diagnosis": r.diagnosis,
            "current_fare": round(r.totalFare, 2), "recommended_fare": round(r.recommended_fare, 2),
            "change_pct": round(r.recommended_change_pct, 1), "estimated_load_factor": round(r.estimated_load_factor, 3),
        }
        for r in top.itertuples()
    ]


PROMPT_TEMPLATE = """You are briefing a revenue-management analyst at the start of the day. \
Below is today's real top-{n} priority queue of flights flagged for review, already ranked \
(diagnosis explains why each was flagged; estimated_load_factor is how full the flight is \
estimated to be, from real aircraft-type seat data, independent of the flag itself).

Data:
{data}

Write a briefing of no more than 120 words, in plain English, for someone who will act on this today. \
Lead with the single most urgent flight and why, in one sentence. Group the rest by cause \
(underpriced/demand surge vs. overpriced/weak demand) rather than listing every flight individually. \
Do not invent any numbers not present in the data. No greeting, no sign-off, no headers."""


def generate_llm(summary: list[dict]) -> tuple[str, str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "no_api_key"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = PROMPT_TEMPLATE.format(n=TOP_N, data=json.dumps(summary, indent=2))
        resp = client.messages.create(model=MODEL, max_tokens=300,
                                       messages=[{"role": "user", "content": prompt}])
        return resp.content[0].text.strip(), "llm"
    except Exception as e:
        return None, f"llm_error: {e}"


def generate_fallback(summary: list[dict]) -> str:
    top = summary[0]
    raises = [s for s in summary if s["change_pct"] > 0]
    cuts = [s for s in summary if s["change_pct"] < 0]
    lines = [f"{top['route']} on {top['flight_date']} is the top priority: {top['diagnosis'].lower()}, "
              f"recommended fare ${top['current_fare']:.0f} -> ${top['recommended_fare']:.0f} "
              f"({top['change_pct']:+.1f}%)."]
    if raises:
        names = ", ".join(f"{s['route']} ({s['change_pct']:+.1f}%)" for s in raises[:4])
        lines.append(f"Raise fares (selling faster than peers, running near-full): {names}.")
    if cuts:
        names = ", ".join(f"{s['route']} ({s['change_pct']:+.1f}%)" for s in cuts[:4])
        lines.append(f"Cut fares (selling slower than peers): {names}.")
    return " ".join(lines)


def run():
    summary = build_queue_summary()
    print(f"=== Daily briefing input: top {TOP_N} priority flights ===")
    print(json.dumps(summary, indent=2))

    text, source = generate_llm(summary)
    if text is None:
        print(f"\n[{source}] -- falling back to template (set ANTHROPIC_API_KEY to use the LLM)")
        text = generate_fallback(summary)
        source = "template_fallback"

    print(f"\n=== Briefing ({source}) ===")
    print(text)

    out = {"source": source, "briefing": text, "data": summary}
    with open(PROCESSED_DIR / "daily_briefing.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to data/processed/daily_briefing.json")
    return out


if __name__ == "__main__":
    run()
