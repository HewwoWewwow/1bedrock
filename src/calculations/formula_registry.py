"""Formula Registry for transparent calculation auditing.

This module provides a central registry of all calculation formulas,
enabling users to understand exactly how each value is computed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum


class FormulaCategory(str, Enum):
    """Categories for organizing formulas."""
    INPUT = "Input"
    DEVELOPMENT = "Development"
    FINANCING = "Financing"
    REVENUE = "Revenue"
    OPERATIONS = "Operations"
    INVESTMENT = "Investment"
    RETURNS = "Returns"


@dataclass
class FormulaDefinition:
    """Definition of a single calculation formula.

    Attributes:
        field_path: Dot-notation path to the field (e.g., "sources_uses.tdc")
        name: Human-readable name (e.g., "Total Development Cost")
        formula: Symbolic formula (e.g., "land + hard_costs + soft_costs + idc")
        inputs: List of input field paths that feed into this formula
        category: Category for grouping formulas
        unit: Display unit (e.g., "$", "%", "units")
        notes: Optional explanation or caveats
    """
    field_path: str
    name: str
    formula: str
    inputs: List[str]
    category: FormulaCategory
    unit: str = "$"
    notes: str = ""


class FormulaRegistry:
    """Central registry of all calculation formulas.

    This singleton class maintains a mapping of field paths to their
    formula definitions, enabling formula lookup and dependency analysis.
    """
    _formulas: Dict[str, FormulaDefinition] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, definition: FormulaDefinition) -> None:
        """Register a formula definition."""
        cls._formulas[definition.field_path] = definition

    @classmethod
    def get(cls, field_path: str) -> Optional[FormulaDefinition]:
        """Get formula definition by field path."""
        cls._ensure_initialized()
        return cls._formulas.get(field_path)

    @classmethod
    def get_all(cls) -> Dict[str, FormulaDefinition]:
        """Get all registered formulas."""
        cls._ensure_initialized()
        return cls._formulas.copy()

    @classmethod
    def get_by_category(cls, category: FormulaCategory) -> List[FormulaDefinition]:
        """Get all formulas in a category."""
        cls._ensure_initialized()
        return [f for f in cls._formulas.values() if f.category == category]

    @classmethod
    def get_inputs(cls, field_path: str) -> List[str]:
        """Get the input field paths for a formula."""
        formula = cls.get(field_path)
        return formula.inputs if formula else []

    @classmethod
    def get_dependents(cls, field_path: str) -> List[str]:
        """Get all formulas that use this field as an input."""
        cls._ensure_initialized()
        dependents = []
        for path, formula in cls._formulas.items():
            if field_path in formula.inputs:
                dependents.append(path)
        return dependents

    @classmethod
    def get_all_ancestors(cls, field_path: str) -> Set[str]:
        """Get all upstream dependencies recursively."""
        cls._ensure_initialized()
        ancestors = set()
        to_process = list(cls.get_inputs(field_path))

        while to_process:
            current = to_process.pop()
            if current not in ancestors:
                ancestors.add(current)
                to_process.extend(cls.get_inputs(current))

        return ancestors

    @classmethod
    def get_all_descendants(cls, field_path: str) -> Set[str]:
        """Get all downstream dependencies recursively."""
        cls._ensure_initialized()
        descendants = set()
        to_process = list(cls.get_dependents(field_path))

        while to_process:
            current = to_process.pop()
            if current not in descendants:
                descendants.add(current)
                to_process.extend(cls.get_dependents(current))

        return descendants

    @classmethod
    def build_dependency_graph(cls):
        """Build a networkx DiGraph of all dependencies.

        Returns:
            nx.DiGraph with nodes for each formula and edges for dependencies.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required for dependency graphs. Install with: pip install networkx")

        cls._ensure_initialized()
        graph = nx.DiGraph()

        # Add all formulas as nodes
        for path, formula in cls._formulas.items():
            graph.add_node(path, **{
                "name": formula.name,
                "category": formula.category.value,
                "formula": formula.formula,
            })

        # Add edges for dependencies
        for path, formula in cls._formulas.items():
            for input_path in formula.inputs:
                graph.add_edge(input_path, path)

        return graph

    @classmethod
    def format_formula_with_values(cls, field_path: str,
                                    values: Dict[str, float]) -> str:
        """Format a formula with actual values substituted.

        Args:
            field_path: The formula to format
            values: Dict mapping input names to their values

        Returns:
            String like "land + hard_costs = $3,000,000 + $31,000,000"
        """
        formula = cls.get(field_path)
        if not formula:
            return ""

        result = formula.formula
        value_parts = []

        for input_name in formula.inputs:
            # Extract the short name from the path
            short_name = input_name.split(".")[-1]
            if input_name in values:
                value_parts.append(f"${values[input_name]:,.0f}")

        if value_parts:
            return f"{formula.formula} = {' + '.join(value_parts)}"
        return formula.formula

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure the registry is populated with formulas."""
        if not cls._initialized:
            _populate_registry()
            cls._initialized = True

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (mainly for testing)."""
        cls._formulas = {}
        cls._initialized = False


def _populate_registry() -> None:
    """Populate the registry with all calculation formulas."""

    # =========================================================================
    # INPUT FIELDS (Raw inputs from ProjectInputs)
    # =========================================================================
    inputs = [
        FormulaDefinition(
            field_path="inputs.land_cost",
            name="Land Cost",
            formula="User input",
            inputs=[],
            category=FormulaCategory.INPUT,
            notes="Total land acquisition cost",
        ),
        FormulaDefinition(
            field_path="inputs.target_units",
            name="Target Units",
            formula="User input",
            inputs=[],
            category=FormulaCategory.INPUT,
            unit="units",
        ),
        FormulaDefinition(
            field_path="inputs.hard_cost_per_unit",
            name="Hard Cost per Unit",
            formula="User input",
            inputs=[],
            category=FormulaCategory.INPUT,
        ),
        FormulaDefinition(
            field_path="inputs.soft_cost_pct",
            name="Soft Cost Percentage",
            formula="User input",
            inputs=[],
            category=FormulaCategory.INPUT,
            unit="%",
        ),
        FormulaDefinition(
            field_path="inputs.construction_ltc",
            name="Construction LTC",
            formula="User input",
            inputs=[],
            category=FormulaCategory.INPUT,
            unit="%",
            notes="Loan-to-cost ratio for construction financing",
        ),
        FormulaDefinition(
            field_path="inputs.construction_rate",
            name="Construction Interest Rate",
            formula="User input",
            inputs=[],
            category=FormulaCategory.INPUT,
            unit="%",
        ),
        FormulaDefinition(
            field_path="inputs.exit_cap_rate",
            name="Exit Cap Rate",
            formula="User input",
            inputs=[],
            category=FormulaCategory.INPUT,
            unit="%",
        ),
        FormulaDefinition(
            field_path="inputs.market_rent_psf",
            name="Market Rent per SF",
            formula="User input",
            inputs=[],
            category=FormulaCategory.INPUT,
            unit="$/SF",
        ),
        FormulaDefinition(
            field_path="inputs.vacancy_rate",
            name="Stabilized Vacancy Rate",
            formula="User input",
            inputs=[],
            category=FormulaCategory.INPUT,
            unit="%",
        ),
    ]

    # =========================================================================
    # DEVELOPMENT / SOURCES & USES
    # =========================================================================
    development = [
        FormulaDefinition(
            field_path="sources_uses.hard_costs",
            name="Hard Costs",
            formula="target_units × hard_cost_per_unit + hard_cost_contingency",
            inputs=[
                "inputs.target_units",
                "inputs.hard_cost_per_unit",
                "sources_uses.hard_cost_contingency",
            ],
            category=FormulaCategory.DEVELOPMENT,
            notes="Total construction hard costs including contingency",
        ),
        FormulaDefinition(
            field_path="sources_uses.hard_cost_contingency",
            name="Hard Cost Contingency",
            formula="target_units × hard_cost_per_unit × contingency_pct",
            inputs=["inputs.target_units", "inputs.hard_cost_per_unit"],
            category=FormulaCategory.DEVELOPMENT,
            notes="Typically 5% of base hard costs",
        ),
        FormulaDefinition(
            field_path="sources_uses.soft_costs",
            name="Soft Costs",
            formula="hard_costs × soft_cost_pct + predev + developer_fee + contingency",
            inputs=[
                "sources_uses.hard_costs",
                "inputs.soft_cost_pct",
                "sources_uses.predevelopment_costs",
                "sources_uses.developer_fee",
                "sources_uses.soft_cost_contingency",
            ],
            category=FormulaCategory.DEVELOPMENT,
        ),
        FormulaDefinition(
            field_path="sources_uses.predevelopment_costs",
            name="Predevelopment Costs",
            formula="hard_costs × predev_pct",
            inputs=["sources_uses.hard_costs"],
            category=FormulaCategory.DEVELOPMENT,
            notes="Architecture, engineering, entitlements, etc.",
        ),
        FormulaDefinition(
            field_path="sources_uses.developer_fee",
            name="Developer Fee",
            formula="hard_costs × developer_fee_pct",
            inputs=["sources_uses.hard_costs"],
            category=FormulaCategory.DEVELOPMENT,
            notes="Fee earned by developer, typically 4-5% of hard costs",
        ),
        FormulaDefinition(
            field_path="sources_uses.idc",
            name="Interest During Construction",
            formula="Σ(monthly_debt_balance × monthly_rate) over construction",
            inputs=[
                "sources_uses.construction_loan",
                "inputs.construction_rate",
                "inputs.construction_months",
            ],
            category=FormulaCategory.DEVELOPMENT,
            notes="Iteratively solved: IDC affects TDC which affects loan which affects IDC",
        ),
        FormulaDefinition(
            field_path="sources_uses.tdc",
            name="Total Development Cost",
            formula="land + hard_costs + soft_costs + idc",
            inputs=[
                "inputs.land_cost",
                "sources_uses.hard_costs",
                "sources_uses.soft_costs",
                "sources_uses.idc",
            ],
            category=FormulaCategory.DEVELOPMENT,
            notes="All-in cost to develop the project",
        ),
        FormulaDefinition(
            field_path="sources_uses.construction_loan",
            name="Construction Loan",
            formula="tdc × construction_ltc",
            inputs=["sources_uses.tdc", "inputs.construction_ltc"],
            category=FormulaCategory.DEVELOPMENT,
        ),
        FormulaDefinition(
            field_path="sources_uses.equity",
            name="Required Equity",
            formula="tdc - construction_loan",
            inputs=["sources_uses.tdc", "sources_uses.construction_loan"],
            category=FormulaCategory.DEVELOPMENT,
            notes="Equity required = TDC × (1 - LTC)",
        ),
    ]

    # =========================================================================
    # FINANCING
    # =========================================================================
    financing = [
        FormulaDefinition(
            field_path="perm_loan.amount",
            name="Permanent Loan Amount",
            formula="min(value × ltv_max, noi_constrained_by_dscr, tdc × ltc_max)",
            inputs=[
                "investment.stabilized_value",
                "inputs.perm_ltv_max",
                "operations.stabilized_noi",
                "inputs.perm_dscr_min",
                "sources_uses.tdc",
            ],
            category=FormulaCategory.FINANCING,
            notes="Most constrained of LTV, DSCR, or LTC determines loan size",
        ),
        FormulaDefinition(
            field_path="perm_loan.monthly_payment",
            name="Monthly Debt Service",
            formula="PMT(rate/12, amort_months, -principal)",
            inputs=[
                "perm_loan.amount",
                "inputs.perm_rate",
                "inputs.perm_amort_years",
            ],
            category=FormulaCategory.FINANCING,
            notes="Standard mortgage amortization formula",
        ),
        FormulaDefinition(
            field_path="perm_loan.actual_dscr",
            name="Actual DSCR",
            formula="stabilized_noi / (monthly_payment × 12)",
            inputs=["operations.stabilized_noi", "perm_loan.monthly_payment"],
            category=FormulaCategory.FINANCING,
            notes="Debt Service Coverage Ratio - must meet minimum",
        ),
        FormulaDefinition(
            field_path="perm_loan.actual_ltv",
            name="Actual LTV",
            formula="loan_amount / stabilized_value",
            inputs=["perm_loan.amount", "investment.stabilized_value"],
            category=FormulaCategory.FINANCING,
            unit="%",
        ),
    ]

    # =========================================================================
    # REVENUE
    # =========================================================================
    revenue = [
        FormulaDefinition(
            field_path="gpr.market_gpr_monthly",
            name="Market GPR (Monthly)",
            formula="Σ(units × gsf × rent_psf) for market-rate units",
            inputs=[
                "inputs.target_units",
                "inputs.market_rent_psf",
                "inputs.affordable_pct",
            ],
            category=FormulaCategory.REVENUE,
        ),
        FormulaDefinition(
            field_path="gpr.affordable_gpr_monthly",
            name="Affordable GPR (Monthly)",
            formula="Σ(units × ami_rent) for affordable units",
            inputs=[
                "inputs.target_units",
                "inputs.affordable_pct",
                "inputs.ami_level",
            ],
            category=FormulaCategory.REVENUE,
            notes="Rents capped by AMI limits from HUD tables",
        ),
        FormulaDefinition(
            field_path="gpr.gpr_total",
            name="Total GPR (Monthly)",
            formula="market_gpr + affordable_gpr",
            inputs=["gpr.market_gpr_monthly", "gpr.affordable_gpr_monthly"],
            category=FormulaCategory.REVENUE,
        ),
    ]

    # =========================================================================
    # OPERATIONS
    # =========================================================================
    operations = [
        FormulaDefinition(
            field_path="operations.leaseup_pct",
            name="Lease-Up Occupancy",
            formula="min(months_since_co × leaseup_pace, max_occupancy)",
            inputs=["inputs.leaseup_pace", "inputs.max_occupancy"],
            category=FormulaCategory.OPERATIONS,
            unit="%",
            notes="Linear ramp from 0% to stabilized occupancy",
        ),
        FormulaDefinition(
            field_path="operations.gpr",
            name="Gross Potential Rent",
            formula="base_gpr × (1 + rent_growth)^years",
            inputs=["gpr.gpr_total", "inputs.market_rent_growth"],
            category=FormulaCategory.OPERATIONS,
            notes="GPR with annual escalation applied",
        ),
        FormulaDefinition(
            field_path="operations.vacancy",
            name="Vacancy Loss",
            formula="gpr × (1 - occupancy_pct)",
            inputs=["operations.gpr", "operations.leaseup_pct"],
            category=FormulaCategory.OPERATIONS,
        ),
        FormulaDefinition(
            field_path="operations.egi",
            name="Effective Gross Income",
            formula="gpr - vacancy",
            inputs=["operations.gpr", "operations.vacancy"],
            category=FormulaCategory.OPERATIONS,
        ),
        FormulaDefinition(
            field_path="operations.management_fee",
            name="Management Fee",
            formula="egi × management_fee_pct",
            inputs=["operations.egi"],
            category=FormulaCategory.OPERATIONS,
            notes="Typically 3-4% of EGI",
        ),
        FormulaDefinition(
            field_path="opex.opex_ex_prop_taxes",
            name="Operating Expenses (ex. Taxes)",
            formula="(utilities + maintenance + insurance + misc) × (1 + opex_growth)^years",
            inputs=["inputs.opex_growth"],
            category=FormulaCategory.OPERATIONS,
        ),
        FormulaDefinition(
            field_path="opex.property_taxes",
            name="Property Taxes",
            formula="assessed_value × total_tax_rate × (1 + tax_growth)^years",
            inputs=["inputs.existing_assessed_value", "inputs.property_tax_growth"],
            category=FormulaCategory.OPERATIONS,
            notes="Tax rate is sum of all taxing authorities",
        ),
        FormulaDefinition(
            field_path="opex.total_opex",
            name="Total Operating Expenses",
            formula="opex_ex_taxes + property_taxes",
            inputs=["opex.opex_ex_prop_taxes", "opex.property_taxes"],
            category=FormulaCategory.OPERATIONS,
        ),
        FormulaDefinition(
            field_path="operations.noi",
            name="Net Operating Income",
            formula="egi - management_fee - opex - property_taxes + tif_reimbursement",
            inputs=[
                "operations.egi",
                "operations.management_fee",
                "opex.opex_ex_prop_taxes",
                "opex.property_taxes",
            ],
            category=FormulaCategory.OPERATIONS,
            notes="NOI is the key operating metric",
        ),
        FormulaDefinition(
            field_path="operations.stabilized_noi",
            name="Stabilized NOI (Annual)",
            formula="monthly_noi × 12 at full occupancy",
            inputs=["operations.noi", "inputs.max_occupancy"],
            category=FormulaCategory.OPERATIONS,
            notes="First full year of stabilized operations",
        ),
    ]

    # =========================================================================
    # INVESTMENT / CASH FLOWS
    # =========================================================================
    investment = [
        FormulaDefinition(
            field_path="investment.dev_cost",
            name="Development Cost (Cash Out)",
            formula="-tdc_draw_this_period",
            inputs=["sources_uses.tdc"],
            category=FormulaCategory.INVESTMENT,
            notes="Negative cash flow during development",
        ),
        FormulaDefinition(
            field_path="investment.stabilized_value",
            name="Stabilized Value",
            formula="stabilized_noi / exit_cap_rate",
            inputs=["operations.stabilized_noi", "inputs.exit_cap_rate"],
            category=FormulaCategory.INVESTMENT,
            notes="Value at stabilization for perm loan sizing",
        ),
        FormulaDefinition(
            field_path="investment.terminal_noi",
            name="Terminal NOI",
            formula="noi in reversion period (forward or trailing-12)",
            inputs=["operations.noi"],
            category=FormulaCategory.INVESTMENT,
            notes="NOI used to calculate sale price",
        ),
        FormulaDefinition(
            field_path="investment.reversion",
            name="Reversion (Sale Proceeds)",
            formula="(terminal_noi / exit_cap_rate) × (1 - selling_costs_pct)",
            inputs=[
                "investment.terminal_noi",
                "inputs.exit_cap_rate",
                "inputs.selling_costs_pct",
            ],
            category=FormulaCategory.INVESTMENT,
            notes="Net proceeds from property sale",
        ),
        FormulaDefinition(
            field_path="investment.unlevered_cf",
            name="Unlevered Cash Flow",
            formula="dev_cost + noi + reserves + reversion",
            inputs=[
                "investment.dev_cost",
                "operations.noi",
                "investment.reserves",
                "investment.reversion",
            ],
            category=FormulaCategory.INVESTMENT,
            notes="Cash flow before debt service",
        ),
        FormulaDefinition(
            field_path="investment.levered_cf",
            name="Levered Cash Flow",
            formula="unlevered_cf + net_debt_cf",
            inputs=["investment.unlevered_cf", "debt.net_debt_cf"],
            category=FormulaCategory.INVESTMENT,
            notes="Cash flow to equity after debt service",
        ),
    ]

    # =========================================================================
    # RETURNS
    # =========================================================================
    returns = [
        FormulaDefinition(
            field_path="returns.unlevered_irr",
            name="Unlevered IRR",
            formula="IRR(unlevered_cf_series)",
            inputs=["investment.unlevered_cf"],
            category=FormulaCategory.RETURNS,
            unit="%",
            notes="Return on total investment (before leverage)",
        ),
        FormulaDefinition(
            field_path="returns.levered_irr",
            name="Levered IRR",
            formula="IRR(levered_cf_series)",
            inputs=["investment.levered_cf"],
            category=FormulaCategory.RETURNS,
            unit="%",
            notes="Return on equity investment (after leverage)",
        ),
        FormulaDefinition(
            field_path="returns.yield_on_cost",
            name="Yield on Cost",
            formula="stabilized_noi / tdc",
            inputs=["operations.stabilized_noi", "sources_uses.tdc"],
            category=FormulaCategory.RETURNS,
            unit="%",
            notes="Development return metric - higher is better",
        ),
        FormulaDefinition(
            field_path="returns.equity_multiple",
            name="Equity Multiple",
            formula="total_distributions / total_equity_invested",
            inputs=["sources_uses.equity"],
            category=FormulaCategory.RETURNS,
            unit="x",
            notes="Total cash returned per dollar invested",
        ),
    ]

    # Register all formulas
    all_formulas = inputs + development + financing + revenue + operations + investment + returns
    for formula in all_formulas:
        FormulaRegistry.register(formula)


# Auto-initialize on import (optional, can also call _ensure_initialized lazily)
# _populate_registry()
