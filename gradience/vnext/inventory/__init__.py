"""Inventory-level artifact builders and formatters."""

from gradience.vnext.inventory.corpus_manifest import CorpusManifest
from gradience.vnext.inventory.merge_neighborhood_report import (
    BoundaryWarning,
    ExcludedAdapter,
    MergeNeighborhoodReport,
    NeighborhoodGroup,
    format_merge_neighborhood_report,
)
from gradience.vnext.inventory.neighborhoods import (
    CompatibilityEdge,
    build_compatibility_matrix,
    suggest_merge_neighborhoods,
)
from gradience.vnext.inventory.portfolio import (
    PortfolioRow,
    PortfolioView,
    format_portfolio_html,
    format_portfolio_table,
    scan_portfolio,
)
from gradience.vnext.inventory.summary import (
    InventorySummary,
    build_inventory_summary,
    format_inventory_summary,
)

__all__ = [
    "CorpusManifest",
    "PortfolioRow",
    "PortfolioView",
    "format_portfolio_html",
    "format_portfolio_table",
    "scan_portfolio",
    "InventorySummary",
    "build_inventory_summary",
    "format_inventory_summary",
    "MergeNeighborhoodReport",
    "NeighborhoodGroup",
    "ExcludedAdapter",
    "BoundaryWarning",
    "format_merge_neighborhood_report",
    "CompatibilityEdge",
    "build_compatibility_matrix",
    "suggest_merge_neighborhoods",
]
