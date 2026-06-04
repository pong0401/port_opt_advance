# AGENTS.md

## Adding a New Strategy

Every time a new strategy is added, do all four items below before considering the handoff complete:

1. Add the strategy to precompute with the same evaluation period/alignment rules as the existing strategies in the same family.
2. Add a standalone latest-weight refresh path that recalculates the latest weights from this repo's current data/cache. Do not depend on static latest-weight files from `dynamic_port_opt` for deployed runtime behavior.
3. Add the latest-weight refresh script and output files to the daily GitHub Action that updates latest weights.
4. Add a detailed user-facing strategy explanation in the app/docs, including the strategy settings, universe, optimizer/model, rebalance/timing rules, daily exposure rules, caps, and where latest weights come from.
5. Make the latest-weight display hide asset rows with portfolio weight below `1%` for every strategy, unless the user explicitly asks to inspect small residual positions. Apply this consistently to all newly added and existing strategy latest-weight tables.
