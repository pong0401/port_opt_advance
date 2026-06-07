# AGENTS.md

## Adding a New Strategy

Every time a new strategy is added, do all four items below before considering the handoff complete:

1. Add the strategy to precompute with the same evaluation period/alignment rules as the existing strategies in the same family.
2. Add a standalone latest-weight refresh path that recalculates the latest weights from this repo's current data/cache. Do not depend on static latest-weight files from `dynamic_port_opt` for deployed runtime behavior.
3. Add the latest-weight refresh script and output files to the daily GitHub Action that updates latest weights.
4. Add a detailed user-facing strategy explanation in the app/docs, including the strategy settings, universe, optimizer/model, rebalance/timing rules, daily exposure rules, caps, and where latest weights come from.
5. Make the latest-weight display hide asset rows with portfolio weight below `1%` for every strategy, unless the user explicitly asks to inspect small residual positions. Apply this consistently to all newly added and existing strategy latest-weight tables.

## Strategy Description Style

When adding or updating user-facing strategy descriptions in the app/docs:

1. Write descriptions as readable bullet sections, not one long paragraph.
2. Separate the explanation into at least:
   - `Strategy setup`
   - `Daily exposure rules`
3. In `Strategy setup`, include the base allocation or sleeve mix, universe, selection rules, optimizer/model, objective, rebalance schedule, caps, and latest-weight source.
4. In `Daily exposure rules`, explicitly state whether daily exposure is used. If it is used, include:
   - signal timing, especially lag-1 or next-session execution
   - each asset/sleeve signal and threshold
   - what exposure becomes when the signal is risk-off
   - whether reduced exposure becomes cash, BIL, another sleeve, or `Cash / Reduced Exposure`
5. Do not duplicate the same explanation in a gray caption and an info box. Prefer the bullet info box for user-facing detail.
6. Keep no-overlay strategies explicit: say that no daily exposure overlay is applied and that sleeves remain active until the next rebalance.

## SET100 Updates

Use `https://www.set.or.th/api/set/index/set100/composition?lang=th` as the source URL when updating the latest SET100 composition.
