"""
The handicapper: a last look at each college bet the way a sharp human
bettor would, with everything the model has (Brian, 2026-11-01: "think like
a human bettor that has all the data you have").

For each bet the picks run is about to make, Claude gets the model's number
and lean, the price, the line the bet was first alerted at, the injury and
situation news already looked up (cfb_news), and what this kind of bet did
in the backtest. It answers bet, half (bet half the stake) or pass, with
the football reason. No web searches: it reuses the news, so a review is
about a cent. Reviews are kept in news.json ("_reviews") so a bet is
reviewed again only when its line, price or news changes. Spend counts
toward cfb_news's monthly stop.

Unbacktested: nobody has shown judgment calls like this beat the close, so
the weekly closing-line review is the judge. HANDICAP = False turns it off.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import pandas as pd

import cfb_news

HANDICAP = True

SYSTEM = """You are a sharp, disciplined college football bettor making the \
final call on bets a model has flagged. You see the model's numbers, the \
price, the news a researcher found (injuries, coaching, motivation, travel, \
weather) and how this kind of bet has done historically.

Think like a professional: the betting market is efficient, so the model's \
edge is small and real only when nothing it can't see undermines it. Pass \
when there is a concrete reason the number is stale or wrong: news the model \
doesn't price (a key player out for the first time on the side we bet, a \
coaching or discipline story, a motivation mismatch, bad weather for an \
over), a line that has already moved past our number, or a spot that has \
historically burned this kind of bet. Bet half when the case is real but \
something cuts against it. Otherwise bet: don't pass on vague feelings, and \
don't invent facts that aren't in what you're given.

Answer with one ```json block and nothing after it:
{"decision": "bet|half|pass", "reason": "<one or two sentences on the football case, plain English, no jargon>"}"""

TRACK = {
    "spreads": "Model spreads with a 1.5+ point lean won 56% against opening lines in "
               "2017-21 and 55% in 2022-25; against closing lines the edge mostly fades, "
               "so betting early matters.",
    "totals": "The totals patterns (big favorite with a low total -> over; totals of 60+ -> "
              "under) won 55-57% in every era since 2012. Our own totals model alone is "
              "about break-even.",
}


def _brief(p, injuries: list, situation: list, first: dict | None) -> str:
    side = f"{p.pick} {p.point:+g}" if p.market == "spreads" else f"{p.pick} {p.point:g}"
    lines = [f"Game: {p.away} at {p.home}, kickoff {pd.Timestamp(p.commence):%a %b %-d %H:%M UTC}",
             f"Bet: {side} ({p.market}) at {int(p.price):+d}, {p.book}",
             f"Model: win chance {p.win_chance:.1%}, expected value {p.ev:+.1%}",
             f"Model's reasoning: {p.why}",
             f"History: {TRACK.get(p.market, '')}"]
    if first:
        lines.append(f"First alerted at {first.get('point')} ({first.get('price')}); "
                     f"the line now is {p.point} ({int(p.price)}).")
    lines.append("Injuries: " + (json.dumps(injuries) if injuries else "none reported"))
    lines.append("Situation: " + (json.dumps(situation) if situation else "nothing reported"))
    return "\n".join(lines)


def _parse(text: str) -> dict:
    m = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S)
    try:
        d = json.loads(m[-1]) if m else {}
    except json.JSONDecodeError:
        d = {}
    dec = str(d.get("decision", "")).lower()
    return {"decision": dec if dec in ("bet", "half", "pass") else "bet",
            "reason": str(d.get("reason", "")).strip()}


def _ask(client, brief: str) -> dict:
    resp = client.beta.messages.create(
        model=cfb_news.MODEL, max_tokens=2000, system=SYSTEM,
        messages=[{"role": "user", "content": brief}],
        output_config={"effort": cfb_news.EFFORT},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    cfb_news._tally(getattr(resp, "usage", None))
    if resp.stop_reason == "refusal":
        return {"decision": "bet", "reason": ""}
    return _parse("".join(b.text for b in resp.content if b.type == "text"))


def review(picks: pd.DataFrame, news: dict, history: list | None = None,
           now=None, client=None) -> pd.DataFrame:
    """picks: the run's bets with injuries/situation attached. Drops passes,
    halves 'half' stakes, and adds the reason to `why`. Never fails the run."""
    if not HANDICAP or not len(picks):
        return picks
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    month = f"{now:%Y-%m}"
    budget = news.get("_budget", {})
    spent = max(budget.get("usd", 0.0) if budget.get("month") == month else 0.0,
                cfb_news.ALREADY_SPENT.get(month, 0.0))
    start = cfb_news.cost()
    first = {}
    for b in history or []:
        first[(b.get("game_id"), b.get("market"), b.get("pick"))] = {
            "point": b.get("first_point", b.get("point")), "price": b.get("first_price", b.get("price"))}
    done = news.setdefault("_reviews", {})
    keep, stakes, whys = [], [], []
    for p in picks.itertuples():
        inj = list(getattr(p, "injuries", []) or [])
        sit = list(getattr(p, "situation", []) or [])
        brief = _brief(p, inj, sit, first.get((getattr(p, "game_id", None), p.market, p.pick)))
        key = hashlib.sha1(brief.encode()).hexdigest()[:16]
        got = done.get(key)
        if got is None:
            if spent + cfb_news.cost() - start >= cfb_news.MONTHLY_BUDGET:
                print("handicap: monthly budget reached; bets go out unreviewed")
                got = None
            else:
                try:
                    if client is None:
                        if not os.environ.get("ANTHROPIC_API_KEY"):
                            print("handicap: no ANTHROPIC_API_KEY, skipping")
                            return picks
                        import anthropic
                        client = anthropic.Anthropic()
                    got = _ask(client, brief)
                    done[key] = {**got, "at": now.isoformat()}
                except Exception as e:          # a review is a check, never a reason to fail
                    print(f"handicap: review failed ({type(e).__name__}: {e})")
                    got = None
        dec = (got or {}).get("decision", "bet")
        reason = (got or {}).get("reason", "")
        print(f"handicap: {p.pick} {p.point} -> {dec}: {reason}")
        keep.append(dec != "pass")
        stakes.append(round(p.stake * (0.5 if dec == "half" else 1.0), 2))
        whys.append(f"{p.why} Handicapper: {reason}" if reason else p.why)
    # keep the review cache to the last 300
    if len(done) > 300:
        for k in sorted(done, key=lambda k: done[k].get("at", ""))[:len(done) - 300]:
            done.pop(k)
    news["_budget"] = {"month": month, "usd": round(spent + cfb_news.cost() - start, 4)}
    out = picks.assign(stake=stakes, why=whys)
    return out[keep]
