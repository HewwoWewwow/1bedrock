# Conversation Log: General Real Estate Financial Model Architecture Decision

**Date:** 2026-02-06
**Participants:** User (spb1), Claude
**Topic:** Whether to rewrite or expand the austin-tif-model into a general-purpose real estate financial modeling tool

## Context

The user has a working Austin TIF model (~17,700 lines of application code) built in Python with Streamlit. The model currently handles single-building, residential-only developments with Austin-specific TIF/incentive logic.

The user wants to build a more general real estate financial model based on OOA&D principles with this hierarchy:

```
Development
  └── Parcel(s)
        └── Building(s)
              └── Use(s)  [residential, retail, commercial, etc.]
                    └── UnitType(s)
                          └── User(s)
```

Key requirements surfaced:
- Each user pays to use the space
- Costs exist at every level of the hierarchy (unit, use, building, parcel, development)
- Revenue types vary by use (rents, NNN leases, percentage rent, etc.)
- Multiple parcels, buildings, and uses per development

## Decision

**Expand the existing model** rather than rewrite from scratch.

### Rationale

1. **Calculation core is well-isolated and reusable.** The `src/calculations/` modules (DCF, debt sizing, draw schedules, revenue, costs, IRR, Monte Carlo — ~6,750 lines) are pure functions with no UI coupling. They operate at what is effectively a "use-level" granularity already.

2. **Architecture doesn't resist the change.** The functional-core/imperative-shell pattern means new hierarchical dataclasses can wrap around existing calculation functions without modifying them. No tangled inheritance or embedded UI logic to fight.

3. **Rewrite re-risks validated math.** The financial calculations have been debugged against real Austin scenarios. Rewriting means re-validating every IRR, draw schedule, debt sizing constraint — months of subtle debugging already completed.

4. **The hierarchy is structural, not computational.** The new requirements change how inputs are organized and how results aggregate, not how the core math works. The existing calculations become leaf-node engines composed by a new orchestration layer.

### What the UI situation is

The UI (~8,200 lines of Streamlit) will need near-total rework regardless of approach, since it's built around the flat single-building model. This is true for both expand and rewrite, so it's not a differentiator.

## Existing Architecture Summary

- **Language:** Python 3.10+ with dataclasses, type hints, enums
- **UI:** Streamlit (fully separated from calculations)
- **Pattern:** Functional core (pure functions, immutable dataclasses) + imperative shell (Streamlit state)
- **Key modules:**
  - `src/models/project.py` — `ProjectInputs` dataclass (~40 fields, flat, single-building)
  - `src/models/lookups.py` — Construction types, AMI rents, tax rates
  - `src/models/incentives.py` — Austin-specific TIF/incentive tiers
  - `src/calculations/dcf.py` — Main DCF engine (`run_dcf()`)
  - `src/calculations/revenue.py` — GPR/EGI calculation
  - `src/calculations/costs.py` — TDC with iterative IDC
  - `src/calculations/debt.py` — Construction + permanent loan sizing
  - `src/calculations/draw_schedule.py` — Equity-first draw waterfall
  - `src/calculations/units.py` — Unit allocation (market/affordable split)
  - `src/calculations/taxes.py` — Property tax by jurisdiction
  - `src/calculations/tif.py` — TIF stream/lump sum valuation
  - `src/calculations/metrics.py` — IRR, equity multiple, yield on cost

## Next Step

A detailed expansion strategy document has been created at `docs/EXPANSION_STRATEGY.md` to guide future implementation sprints.
