# Expansion Strategy: General Real Estate Financial Model

**Created:** 2026-02-06
**Purpose:** Guide for Claude (or any implementer) to expand the austin-tif-model into a general-purpose, multi-parcel, multi-building, multi-use real estate financial modeling engine.

**Decision:** Expand the existing codebase. Do NOT rewrite from scratch. The calculation core is validated and well-isolated. See `docs/CONVERSATION_LOG_2026-02-06.md` for full rationale.

---

## 1. TARGET DATA MODEL

The general model follows an OOA&D hierarchy where each level can contain one or more children:

```
Development
  ├── development-level costs (entitlements, master planning, infrastructure)
  ├── development-level financing (if applicable)
  │
  └── Parcel(s)
        ├── parcel-level costs (land acquisition, site work, utilities)
        ├── parcel-level attributes (acreage, zoning, existing assessed value)
        │
        └── Building(s)
              ├── building-level costs (structural, core/shell, parking, common areas)
              ├── building-level attributes (construction type, floors, GSF)
              │
              └── Use(s)  [residential, retail, office, industrial, parking, etc.]
                    ├── use-level costs (TI, specialized MEP, lease commissions)
                    ├── use-level operating assumptions (vacancy, opex, growth rates)
                    │
                    └── UnitType(s)
                          ├── unit-type attributes (SF, count, rent/rate)
                          │
                          └── User(s) / Tenant(s)
                                └── lease terms, rent, escalations, TI, free rent
```

### 1.1 New Dataclasses to Create

Create these in a new file `src/models/hierarchy.py`:

```python
@dataclass
class Development:
    """Top-level entity. A development project that may span multiple parcels."""
    name: str
    parcels: list[Parcel]
    timing: DevelopmentTiming  # predevelopment, construction, leaseup, operations months
    financing: DevelopmentFinancing  # construction loan, perm loan params
    costs: DevelopmentCosts  # development-level costs only
    exit_assumptions: ExitAssumptions  # cap rate, selling costs, NOI basis

@dataclass
class Parcel:
    """A discrete piece of land within a development."""
    name: str
    buildings: list[Building]
    land_cost: float
    acreage: float
    existing_assessed_value: float
    tax_rates: dict[str, float]  # jurisdiction -> rate per $100
    costs: ParcelCosts  # site work, utilities, off-sites

@dataclass
class Building:
    """A physical structure on a parcel."""
    name: str
    uses: list[Use]
    construction_type: ConstructionType
    total_gsf: int
    floors: int
    costs: BuildingCosts  # core/shell, parking structure, common areas, vertical transport

@dataclass
class Use:
    """A use within a building (residential, retail, office, etc.)."""
    name: str
    use_type: UseType  # enum: RESIDENTIAL, RETAIL, OFFICE, INDUSTRIAL, PARKING, etc.
    unit_types: list[UnitType]
    gsf: int  # gross SF allocated to this use
    efficiency: float  # net-to-gross ratio
    costs: UseCosts  # TI, specialized systems, lease commissions
    operating: UseOperating  # vacancy, opex, mgmt fee, reserves, growth rates
    revenue_model: RevenueModel  # how revenue is calculated for this use type

@dataclass
class UnitType:
    """A category of leasable space within a use."""
    name: str  # e.g., "1BR", "Inline Retail", "Class A Office"
    nsf: int  # net square feet per unit
    count: int  # number of units of this type
    base_rent: float  # monthly rent (residential) or annual rent PSF (commercial)
    users: list[User]  # optional: individual tenant detail

@dataclass
class User:
    """An occupant of a unit. Optional granularity — can be omitted for pro forma."""
    lease_start_month: int
    lease_term_months: int
    rent_monthly: float
    annual_escalation: float
    free_rent_months: int = 0
    ti_allowance_psf: float = 0.0
```

### 1.2 Revenue Model by Use Type

Different uses generate revenue differently. Create `src/models/revenue_models.py`:

```python
class UseType(Enum):
    RESIDENTIAL = "residential"
    RETAIL = "retail"
    OFFICE = "office"
    INDUSTRIAL = "industrial"
    PARKING = "parking"
    HOTEL = "hotel"

@dataclass
class RevenueModel:
    """Defines how revenue is calculated for a use."""
    use_type: UseType
    # Residential: monthly rent per unit
    # Retail: annual rent PSF (NNN or gross), percentage rent threshold
    # Office: annual rent PSF (full service or NNN)
    # Parking: monthly rate per space
    # Hotel: ADR x occupancy x rooms
    rent_basis: RentBasis  # MONTHLY_PER_UNIT, ANNUAL_PSF, MONTHLY_PER_SPACE, ADR
    lease_type: LeaseType  # GROSS, NNN, MODIFIED_GROSS, PERCENTAGE
    reimbursement_pct: float = 0.0  # For NNN: what % of opex is reimbursed by tenant
    percentage_rent_threshold: float = 0.0  # For retail percentage rent
    percentage_rent_rate: float = 0.0
```

### 1.3 Cost Allocation Hierarchy

Create `src/models/cost_hierarchy.py`:

```python
@dataclass
class DevelopmentCosts:
    """Costs at the development level (spread across all parcels)."""
    entitlements: float = 0.0
    master_planning: float = 0.0
    legal: float = 0.0
    infrastructure: float = 0.0  # shared roads, utilities
    development_fee: float = 0.0

@dataclass
class ParcelCosts:
    """Costs at the parcel level."""
    site_work: float = 0.0
    utilities: float = 0.0
    offsite_improvements: float = 0.0
    impact_fees: float = 0.0
    environmental: float = 0.0

@dataclass
class BuildingCosts:
    """Costs at the building level."""
    hard_cost_psf: float = 0.0  # core/shell cost per GSF
    parking_cost: float = 0.0
    common_area_cost: float = 0.0
    vertical_transport: float = 0.0  # elevators
    soft_cost_pct: float = 0.30  # as % of hard costs

@dataclass
class UseCosts:
    """Costs at the use level."""
    tenant_improvements_psf: float = 0.0
    specialized_mep_psf: float = 0.0
    lease_commissions_pct: float = 0.0  # as % of lease value
    fit_out_psf: float = 0.0
```

---

## 2. MODULE-BY-MODULE EXPANSION PLAN

This section maps each existing module to its role in the expanded model, specifying what changes (if any) are needed.

### 2.1 Modules That Can Be Used AS-IS (no changes)

| Module | Why It Works |
|--------|-------------|
| `calculations/debt.py` | `size_construction_loan()` and `size_permanent_loan()` take scalar inputs (TDC, NOI, rates). They don't know or care about hierarchy. Call them with rolled-up values. |
| `calculations/draw_schedule.py` | `generate_draw_schedule()` takes TDC, equity, loan amount, and timing. Feed it development-level totals. Works unchanged. |
| `calculations/taxes.py` | `calculate_property_taxes()` takes assessed value, existing value, and tax rates. Call it per-parcel, then sum. Works unchanged. |
| `calculations/tif.py` | TIF valuation is Austin-specific but self-contained. Can be called per-parcel or made optional. No changes needed. |
| `models/incentives.py` | Austin-specific. Keep as-is. Future markets get their own incentive modules following the same pattern. |

### 2.2 Modules That Need MINOR Adaptation

| Module | Current State | Change Needed |
|--------|--------------|---------------|
| `calculations/revenue.py` | `calculate_gpr()` takes `list[UnitAllocation]` and sums market + affordable rent. Returns `GPRResult`. | **Generalize to handle non-residential revenue.** Add a `calculate_use_revenue()` function that dispatches by `UseType`. Residential path calls existing `calculate_gpr()`. Retail/office/etc. get new sibling functions. All return a common `RevenueResult` shape. |
| `calculations/costs.py` | `calculate_tdc()` takes flat cost inputs. `calculate_idc()` is pure math. | **Keep `calculate_idc()` unchanged.** Replace `calculate_tdc()` with a `calculate_tdc_from_hierarchy()` that sums costs from all hierarchy levels, then calls `calculate_idc()` with the total. The existing `calculate_tdc()` can remain as a convenience for flat/simple cases. |
| `calculations/units.py` | `allocate_units()` handles residential market/affordable split. | **Keep for residential use type.** Non-residential uses don't need unit allocation — they have `UnitType.count` directly. The residential path continues to use this function. |

### 2.3 Modules That Need SIGNIFICANT Rework

| Module | Current State | Change Needed |
|--------|--------------|---------------|
| `calculations/dcf.py` | `run_dcf()` is a 400-line monolith that orchestrates everything for a single flat project. Hardcoded to residential assumptions. | **This is the main expansion target.** See Section 3 below for the detailed orchestration redesign. |
| `models/project.py` | `ProjectInputs` is a flat 40-field dataclass — the antithesis of the hierarchy. | **Deprecate as the top-level input.** Replace with `Development` as the entry point. `ProjectInputs` can remain as a "flat mode" convenience that auto-constructs a single-parcel, single-building, single-use `Development` for backward compatibility. |

### 2.4 New Modules to Create

| Module | Purpose |
|--------|---------|
| `models/hierarchy.py` | The dataclasses from Section 1.1 (Development, Parcel, Building, Use, UnitType, User) |
| `models/revenue_models.py` | UseType enum, RevenueModel, RentBasis, LeaseType |
| `models/cost_hierarchy.py` | DevelopmentCosts, ParcelCosts, BuildingCosts, UseCosts |
| `calculations/orchestrator.py` | New top-level engine that walks the hierarchy and composes existing calculation functions. See Section 3. |
| `calculations/rollup.py` | Aggregation logic: sum Use results → Building, Building results → Parcel, Parcel results → Development |
| `calculations/revenue_commercial.py` | Revenue functions for retail, office, industrial, parking, hotel |
| `calculations/adapters.py` | Functions that convert hierarchy-level inputs into the flat formats existing calculation functions expect |

---

## 3. THE ORCHESTRATOR: How `run_dcf()` Gets Replaced

The current `run_dcf()` does everything in sequence for one flat project. The new orchestrator decomposes this into a tree walk:

```
run_development_dcf(development: Development) -> DevelopmentResult
    │
    ├── For each Parcel:
    │     ├── Calculate parcel-level costs (land, site work)
    │     ├── Calculate property taxes (per-parcel assessed value & tax rates)
    │     │
    │     ├── For each Building:
    │     │     ├── Calculate building-level costs (hard costs from construction type)
    │     │     │
    │     │     ├── For each Use:
    │     │     │     ├── Calculate use-level revenue (dispatch by UseType)
    │     │     │     ├── Calculate use-level costs (TI, specialized MEP)
    │     │     │     ├── Calculate use-level opex (vacancy, mgmt, reserves)
    │     │     │     └── Return UseResult (GPR, EGI, NOI, costs)
    │     │     │
    │     │     └── Roll up: BuildingResult = sum(UseResults) + building costs
    │     │
    │     └── Roll up: ParcelResult = sum(BuildingResults) + parcel costs + taxes
    │
    ├── Roll up: DevelopmentResult = sum(ParcelResults) + development costs
    │
    ├── Calculate TDC (sum of all cost levels + IDC)  ← uses existing calculate_idc()
    ├── Size construction loan                         ← uses existing size_construction_loan()
    ├── Size permanent loan                            ← uses existing size_permanent_loan()
    ├── Generate draw schedule                         ← uses existing generate_draw_schedule()
    ├── Build monthly cash flows                       ← rewritten loop, but same Phase logic
    ├── Calculate IRR, equity multiple                 ← uses existing npf.irr() pattern
    │
    └── Return DevelopmentResult with full cash flows and metrics
```

### 3.1 Key Design Principle: Use-Level Is the Atomic Revenue Unit

Every Use produces a `UseResult` that contains:
- Monthly GPR (how this use generates revenue)
- Vacancy rate (how much is lost)
- Monthly EGI
- Monthly OpEx (use-level only)
- Monthly NOI contribution (before building/parcel/development costs)

The orchestrator sums these up the tree, then applies higher-level costs at each aggregation step.

### 3.2 Key Design Principle: Costs Flow Down, Revenue Flows Up

```
                    Revenue flows UP
                         ▲
Development ────► dev costs applied here
    │                    ▲
    Parcel ──────► parcel costs + taxes applied here
        │                ▲
        Building ──► building costs applied here
            │            ▲
            Use ─────► revenue generated here, use costs applied here
```

### 3.3 Monthly Cash Flow Loop

The monthly cash flow loop in the current `run_dcf()` (lines 302-457) stays structurally the same — it iterates over months and switches on Phase. The key change is that revenue and opex come from aggregated Use-level results instead of a single flat GPR calculation:

```python
# CURRENT (flat):
cf.gross_potential_rent = gpr_result.market_gpr_monthly * growth + gpr_result.affordable_gpr_monthly * growth

# NEW (hierarchical):
cf.gross_potential_rent = sum(
    use_result.gpr_monthly * use_result.get_growth_factor(years_operating)
    for use_result in all_use_results
)
```

The same pattern applies to opex, vacancy, etc.

---

## 4. BACKWARD COMPATIBILITY

The Austin TIF model must continue to work. Strategy:

1. **Keep `ProjectInputs`** but add a `.to_development()` method that creates a single-parcel, single-building, single-use Development:
   ```python
   def to_development(self) -> Development:
       """Convert flat ProjectInputs to hierarchical Development."""
       use = Use(
           name="Residential",
           use_type=UseType.RESIDENTIAL,
           unit_types=[...from self.unit_mix...],
           # ...
       )
       building = Building(name="Main", uses=[use], construction_type=self.construction_type, ...)
       parcel = Parcel(name="Site", buildings=[building], land_cost=..., ...)
       return Development(name="Project", parcels=[parcel], ...)
   ```

2. **Keep `run_dcf()`** as a convenience wrapper:
   ```python
   def run_dcf(inputs: ProjectInputs, scenario: Scenario) -> DCFResult:
       """Legacy entry point. Converts to hierarchy and delegates."""
       development = inputs.to_development()
       dev_result = run_development_dcf(development)
       return dev_result.to_legacy_dcf_result()
   ```

3. **All existing tests continue to pass** against the legacy `run_dcf()` interface.

---

## 5. PHASED IMPLEMENTATION ORDER

### Phase 1: Data Model Foundation
**Goal:** Define the hierarchy without changing any calculations.

- [ ] Create `src/models/hierarchy.py` with Development, Parcel, Building, Use, UnitType, User dataclasses
- [ ] Create `src/models/revenue_models.py` with UseType, RevenueModel, RentBasis, LeaseType
- [ ] Create `src/models/cost_hierarchy.py` with cost dataclasses at each level
- [ ] Add `ProjectInputs.to_development()` adapter method
- [ ] Write tests that verify `to_development()` round-trips correctly

### Phase 2: Revenue Generalization
**Goal:** Support non-residential revenue without breaking residential.

- [ ] Create `src/calculations/revenue_commercial.py` with functions for retail, office, industrial, parking revenue
- [ ] Add `calculate_use_revenue(use: Use) -> UseRevenueResult` dispatcher that calls existing `calculate_gpr()` for residential and new functions for others
- [ ] Define `UseRevenueResult` as a common return type
- [ ] Write tests for each revenue type in isolation

### Phase 3: Cost Hierarchy & Roll-Up
**Goal:** Sum costs from all hierarchy levels into a TDC.

- [ ] Create `src/calculations/rollup.py` with functions to aggregate costs up the tree
- [ ] Create `src/calculations/adapters.py` to convert hierarchy totals into the format `calculate_idc()` expects
- [ ] Write `calculate_tdc_from_hierarchy(development: Development) -> TDCResult` that sums all cost levels, then calls existing `calculate_idc()`
- [ ] Test that a single-parcel/single-building/single-use Development produces the same TDC as the flat `calculate_tdc()`

### Phase 4: Orchestrator
**Goal:** Replace the monolithic `run_dcf()` with a tree-walking orchestrator.

- [ ] Create `src/calculations/orchestrator.py` with `run_development_dcf(development: Development) -> DevelopmentResult`
- [ ] Implement the tree walk from Section 3
- [ ] Wire in existing debt sizing, draw schedule, and tax functions unchanged
- [ ] Implement the monthly cash flow loop using aggregated Use-level results
- [ ] Add `DevelopmentResult.to_legacy_dcf_result()` for backward compatibility
- [ ] **Critical test:** Run the Austin TIF scenarios through both old `run_dcf()` and new `run_development_dcf()` (via `to_development()`) and verify identical outputs to within floating-point tolerance

### Phase 5: Multi-Entity Support
**Goal:** Actually support multiple parcels, buildings, and uses.

- [ ] Add phasing/timing logic (buildings can start construction at different times)
- [ ] Add cross-building financing (single construction loan for whole development vs. per-building)
- [ ] Add weighted-average exit cap rate logic (blended cap rate across uses)
- [ ] Add per-parcel tax calculations that roll up correctly
- [ ] Write integration tests with a mixed-use scenario (e.g., podium retail + residential tower)

### Phase 6: Austin Specialization as Plugin
**Goal:** Make Austin-specific logic (TIF, SMART waivers, incentive tiers) a pluggable module.

- [ ] Move Austin-specific incentive logic into `src/plugins/austin/` or `src/markets/austin/`
- [ ] Define an `IncentivePlugin` protocol/interface that the orchestrator can call
- [ ] Other markets can implement their own incentive plugins following the same pattern
- [ ] Keep AMI rent tables as market-specific lookup data

---

## 6. RESULT HIERARCHY

The output should mirror the input hierarchy:

```python
@dataclass
class DevelopmentResult:
    """Top-level output."""
    parcel_results: list[ParcelResult]

    # Rolled-up metrics
    tdc: float
    equity_required: float
    levered_irr: float
    unlevered_irr: float
    equity_multiple: float
    yield_on_cost: float
    stabilized_noi: float

    # Monthly cash flows (development-level)
    monthly_cash_flows: list[MonthlyCashFlow]

    # Sources & uses
    draw_schedule: DrawSchedule

    def to_legacy_dcf_result(self) -> DCFResult:
        """Convert to legacy DCFResult for backward compatibility."""

@dataclass
class ParcelResult:
    building_results: list[BuildingResult]
    land_cost: float
    property_taxes_annual: float
    parcel_noi_contribution: float

@dataclass
class BuildingResult:
    use_results: list[UseResult]
    building_costs: float
    building_noi_contribution: float

@dataclass
class UseResult:
    use_type: UseType
    gpr_monthly: float
    vacancy_rate: float
    egi_monthly: float
    opex_monthly: float
    noi_monthly: float
    unit_count: int
    nsf_total: int
```

---

## 7. WHAT NOT TO CHANGE

These things should be preserved exactly:

1. **`calculate_idc()` in costs.py** — The iterative IDC convergence logic is correct and general. Just feed it summed costs.
2. **`size_construction_loan()` and `size_permanent_loan()` in debt.py** — Pure scalar math. Feed rolled-up values.
3. **`generate_draw_schedule()` in draw_schedule.py** — Takes TDC, equity, loan, timing. Feed development-level totals.
4. **The Phase enum and monthly loop structure in dcf.py** — The 5-phase model (predevelopment → construction → leaseup → operations → reversion) is general to all real estate development. Keep it.
5. **IRR calculation pattern** — Building arrays of monthly cash flows and calling `npf.irr()` is correct.
6. **Property tax calculation in taxes.py** — Per-jurisdiction, per-$100 assessed value. Call per-parcel.

---

## 8. TESTING STRATEGY

### The Golden Rule
After each phase, the Austin TIF model scenarios must produce **identical results** (within floating-point tolerance) through both the old flat path and the new hierarchical path. This is the regression test that proves the expansion didn't break the validated math.

### Test Fixture
Create a canonical test fixture:
```python
# tests/fixtures/mixed_use_development.py
def create_test_mixed_use() -> Development:
    """2-parcel, 3-building, 4-use test development."""
    # Parcel A: Residential tower (300 units) + ground-floor retail (15,000 SF)
    # Parcel B: Office building (80,000 SF) + structured parking (400 spaces)
    ...
```

### Per-Phase Tests
- **Phase 1:** `to_development()` round-trip
- **Phase 2:** Revenue calculations per use type match hand calculations
- **Phase 3:** TDC from hierarchy matches flat TDC for single-building residential
- **Phase 4:** Full DCF through hierarchy matches legacy `run_dcf()` for Austin scenarios
- **Phase 5:** Multi-building scenario produces reasonable metrics (sanity checks against industry benchmarks)

---

## 9. FILE STRUCTURE AFTER EXPANSION

```
src/
├── models/
│   ├── project.py           # KEEP - legacy flat model + to_development() adapter
│   ├── hierarchy.py          # NEW  - Development, Parcel, Building, Use, UnitType, User
│   ├── revenue_models.py     # NEW  - UseType, RevenueModel, RentBasis, LeaseType
│   ├── cost_hierarchy.py     # NEW  - cost dataclasses at each level
│   ├── lookups.py            # KEEP - construction types, unit defaults
│   ├── incentives.py         # KEEP - Austin-specific (future: move to plugins)
│   └── scenario_config.py    # KEEP
│
├── calculations/
│   ├── orchestrator.py       # NEW  - run_development_dcf(), tree walk
│   ├── rollup.py             # NEW  - aggregate results up the hierarchy
│   ├── adapters.py           # NEW  - convert hierarchy inputs to flat formats
│   ├── revenue_commercial.py # NEW  - retail, office, industrial, parking revenue
│   ├── dcf.py                # KEEP - legacy run_dcf() wraps orchestrator
│   ├── revenue.py            # KEEP - residential GPR/EGI (called by orchestrator)
│   ├── costs.py              # KEEP - calculate_idc(), calculate_tdc() (legacy)
│   ├── debt.py               # KEEP - unchanged
│   ├── draw_schedule.py      # KEEP - unchanged
│   ├── taxes.py              # KEEP - unchanged
│   ├── tif.py                # KEEP - unchanged
│   ├── units.py              # KEEP - residential unit allocation
│   ├── metrics.py            # KEEP - unchanged
│   ├── sources_uses.py       # KEEP - may need hierarchy-aware version later
│   ├── monte_carlo.py        # KEEP - parameterize over hierarchy inputs later
│   └── detailed_cashflow.py  # KEEP - may need hierarchy-aware version later
│
└── scenarios.py              # KEEP - extend to accept Development inputs
```

---

## 10. INSTRUCTIONS FOR CLAUDE IN FUTURE SPRINTS

When implementing any phase of this expansion:

1. **Read this document first.** It is the authoritative source for architectural decisions.
2. **Read `docs/CONVERSATION_LOG_2026-02-06.md`** for the rationale behind the expand-vs-rewrite decision.
3. **Do not modify existing calculation functions** unless this document explicitly says to. The strategy is to compose them, not rewrite them.
4. **Run the existing test suite after every change** to verify backward compatibility.
5. **Follow the existing code style:** dataclasses for data, pure functions for calculations, type hints everywhere, enums for domain concepts.
6. **Each new module should be independently testable** with no Streamlit or UI dependencies.
7. **The Phase 4 regression test is the most important test in the project.** If the hierarchical path doesn't produce identical results to the flat path for Austin scenarios, something is wrong. Do not proceed to Phase 5 until this passes.
8. **When in doubt about financial modeling concepts** (cap rates, DSCR, NNN leases, etc.), ask the user rather than guessing. These are domain-specific terms with precise meanings.
