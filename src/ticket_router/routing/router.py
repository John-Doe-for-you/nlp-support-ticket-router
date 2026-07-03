"""Category-to-team routing.

Public API
----------
* ``CATEGORY_TO_TEAM`` : the locked mapping (PROJECT_PLAN §8).
* ``DEFAULT_TEAM``     : fallback team for unknown categories.
* ``ALL_TEAMS``        : the set of team names reachable via the router.
* ``route_ticket(category)`` : module-level convenience lookup.
* ``Router``           : thin OO wrapper exposing the same lookup with a
                         configurable default team and an `is_known`
                         helper for tests / debugging.

The router is intentionally tiny. It contains no model code; it just
turns the category label produced by the Day 7 classifier into the team
the Day 15 API will route the ticket to. Keeping it isolated makes it
trivial to unit-test and easy to extend (e.g. with weighted routing or
per-team capacity) in a later phase without touching the rest of the
pipeline.
"""

from __future__ import annotations

CATEGORY_TO_TEAM: dict[str, str] = {
    "Billing": "billing-team",
    "Authentication": "identity-team",
    "Bug Report": "engineering-team",
    "Feature Request": "product-team",
    "Technical Setup": "support-team",
}

# Fallback team for any category we don't recognize. Picked so the
# /classify endpoint never 500s on novel input - the worst case is a
# ticket lands in a general queue rather than a specialist one.
DEFAULT_TEAM: str = "support-team"

# All team names reachable through the router. Useful for the Day 16
# stats endpoint and for assertion in tests.
ALL_TEAMS: frozenset[str] = frozenset(CATEGORY_TO_TEAM.values()) | {DEFAULT_TEAM}


def route_ticket(category: str) -> str:
    """Return the team responsible for ``category``.

    Unknown categories fall back to ``DEFAULT_TEAM`` so the API never
    500s on novel input.
    """
    if not isinstance(category, str):
        return DEFAULT_TEAM
    return CATEGORY_TO_TEAM.get(category, DEFAULT_TEAM)


class Router:
    """Category -> team lookup with a configurable fallback.

    Construction is cheap; the router is safe to share across threads.
    For request paths, prefer the module-level :func:`route_ticket`
    helper (which uses the locked defaults) or instantiate a
    ``Router()`` once and reuse it.
    """

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        default_team: str = DEFAULT_TEAM,
    ) -> None:
        self._mapping: dict[str, str] = (
            dict(CATEGORY_TO_TEAM) if mapping is None else dict(mapping)
        )
        self.default_team: str = str(default_team)

    @property
    def mapping(self) -> dict[str, str]:
        """The active category -> team table (read-only copy semantics)."""
        return dict(self._mapping)

    @property
    def teams(self) -> frozenset[str]:
        """The set of team names reachable through this router."""
        return frozenset(self._mapping.values()) | {self.default_team}

    def is_known_category(self, category: str) -> bool:
        """Return True if ``category`` has an explicit team mapping."""
        return isinstance(category, str) and category in self._mapping

    def route(self, category: str) -> str:
        """Look up the team for ``category``; fall back if unknown."""
        if not isinstance(category, str):
            return self.default_team
        return self._mapping.get(category, self.default_team)

    def route_with_reason(self, category: str) -> tuple[str, bool]:
        """Return ``(team, is_exact_match)``.

        ``is_exact_match`` is True when the category was found in the
        mapping, False when the default was used. Useful for the
        Day 18 edge-case analysis to flag tickets that landed in a
        fallback queue.
        """
        if isinstance(category, str) and category in self._mapping:
            return self._mapping[category], True
        return self.default_team, False


__all__ = [
    "CATEGORY_TO_TEAM",
    "DEFAULT_TEAM",
    "ALL_TEAMS",
    "route_ticket",
    "Router",
]
