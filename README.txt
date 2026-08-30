CFB v0.4.1 fixes historical odds rows where missing moneyline prices are returned as 0, which caused a division-by-zero during backtests. Invalid/missing American odds are now skipped defensively.
