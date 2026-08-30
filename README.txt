CFB Edge v1.6.0-APP-POLISH

This is a UI/product release. Model logic is unchanged.

What's new
==========
- Team logos from the CFBD FBS team data already loaded by the app.
- Top 5/10 bet cards show the relevant team logo.
- Totals show both matchup logos.
- Team initials are used when a logo is unavailable.
- Premium CFB EDGE slate header with date, market coverage and live-model indicator.
- Game dropdowns remain chronological by kickoff.
- Inside each game, an away/home logo matchup header appears before the ranked markets.
- Cleaner mobile spacing and more app-like expander styling.

Workflow
========
1. Run slate.
2. Top 5 / Top 10 = quickest actionable betting card.
3. Next Best Leans = optional lower-grade ideas.
4. All Games = chronological navigation.
5. Open matchup = top bet first, all other markets ranked below it.

Model protections preserved
===========================
- FCS fallback guard
- Extreme longshot ML block
- +500 to +999 ML lean ceiling
- A/B-only official Top Bets
- Cross-market spread / ML / total ranking
