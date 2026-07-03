"""Tests for the Day 11 category-to-team router.

The mapping is locked in ``docs/PROJECT_PLAN.md`` §8. These tests cover:

* the module-level mapping constants (``CATEGORY_TO_TEAM``, ``DEFAULT_TEAM``,
  ``ALL_TEAMS``)
* the :func:`route_ticket` convenience helper
* the :class:`Router` class (lookup, fallback, custom mapping, helper methods)
"""

from __future__ import annotations

import pytest

from ticket_router.routing.router import (
    ALL_TEAMS,
    CATEGORY_TO_TEAM,
    DEFAULT_TEAM,
    Router,
    route_ticket,
)


# ---------------------------------------------------------------------------
# Constants / shape
# ---------------------------------------------------------------------------


def test_category_to_team_has_all_five_categories() -> None:
    expected = {
        "Billing",
        "Authentication",
        "Bug Report",
        "Feature Request",
        "Technical Setup",
    }
    assert set(CATEGORY_TO_TEAM) == expected


def test_category_to_team_matches_plan_section_8() -> None:
    """The plan's routing table is exactly this dict."""
    assert CATEGORY_TO_TEAM == {
        "Billing": "billing-team",
        "Authentication": "identity-team",
        "Bug Report": "engineering-team",
        "Feature Request": "product-team",
        "Technical Setup": "support-team",
    }


def test_default_team_is_support_team() -> None:
    assert DEFAULT_TEAM == "support-team"


def test_all_teams_includes_default() -> None:
    assert DEFAULT_TEAM in ALL_TEAMS
    # The default must be a real team string, not a placeholder.
    assert isinstance(DEFAULT_TEAM, str) and DEFAULT_TEAM


def test_all_teams_includes_every_mapped_team() -> None:
    for team in CATEGORY_TO_TEAM.values():
        assert team in ALL_TEAMS


def test_all_team_values_are_unique() -> None:
    # No two categories should map to the same team in the default table.
    teams = list(CATEGORY_TO_TEAM.values())
    assert len(teams) == len(set(teams))


# ---------------------------------------------------------------------------
# Module-level route_ticket
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category,expected",
    [
        ("Billing", "billing-team"),
        ("Authentication", "identity-team"),
        ("Bug Report", "engineering-team"),
        ("Feature Request", "product-team"),
        ("Technical Setup", "support-team"),
    ],
)
def test_route_ticket_known_categories(category: str, expected: str) -> None:
    assert route_ticket(category) == expected


def test_route_ticket_unknown_category_falls_back() -> None:
    assert route_ticket("nonsense-category") == DEFAULT_TEAM


def test_route_ticket_empty_string_falls_back() -> None:
    assert route_ticket("") == DEFAULT_TEAM


def test_route_ticket_none_returns_default() -> None:
    # Defensive: a None input should never crash the API.
    assert route_ticket(None) == DEFAULT_TEAM  # type: ignore[arg-type]


def test_route_ticket_non_string_returns_default() -> None:
    assert route_ticket(123) == DEFAULT_TEAM  # type: ignore[arg-type]
    assert route_ticket(["Billing"]) == DEFAULT_TEAM  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Router class
# ---------------------------------------------------------------------------


@pytest.fixture()
def router() -> Router:
    return Router()


def test_router_uses_default_mapping(router: Router) -> None:
    assert router.mapping == CATEGORY_TO_TEAM


def test_router_default_team(router: Router) -> None:
    assert router.default_team == DEFAULT_TEAM


def test_router_route_known(router: Router) -> None:
    assert router.route("Billing") == "billing-team"
    assert router.route("Authentication") == "identity-team"


def test_router_route_unknown_falls_back(router: Router) -> None:
    assert router.route("Made Up Category") == DEFAULT_TEAM
    assert router.route("") == DEFAULT_TEAM
    assert router.route(None) == DEFAULT_TEAM  # type: ignore[arg-type]


def test_router_mapping_is_a_copy(router: Router) -> None:
    # Mutating the returned dict must not change the router's internal state.
    snap = router.mapping
    snap["Billing"] = "rogue-team"
    assert router.mapping["Billing"] == "billing-team"
    assert router.route("Billing") == "billing-team"


def test_router_constructor_does_not_share_state() -> None:
    # Constructor must copy the input dict so two routers stay independent.
    a = Router({"X": "team-a"})
    b = Router({"X": "team-a"})
    a.mapping["Y"] = "team-b"  # type: ignore[index]
    assert "Y" not in b.mapping


def test_router_custom_mapping_overrides_default() -> None:
    r = Router({"Billing": "custom-finance-team"})
    assert r.route("Billing") == "custom-finance-team"
    # Categories not in the custom map still hit the default.
    assert r.route("Authentication") == DEFAULT_TEAM


def test_router_custom_default_team() -> None:
    r = Router(default_team="triage-team")
    assert r.route("anything-unknown") == "triage-team"
    assert r.default_team == "triage-team"


def test_router_teams_includes_default() -> None:
    r = Router({"Billing": "billing-team"}, default_team="triage-team")
    assert "billing-team" in r.teams
    assert "triage-team" in r.teams
    assert r.teams == frozenset({"billing-team", "triage-team"})


def test_router_is_known_category(router: Router) -> None:
    assert router.is_known_category("Billing") is True
    assert router.is_known_category("Feature Request") is True
    assert router.is_known_category("Not A Category") is False
    assert router.is_known_category("") is False


def test_router_is_known_category_handles_non_string(router: Router) -> None:
    assert router.is_known_category(None) is False  # type: ignore[arg-type]
    assert router.is_known_category(42) is False  # type: ignore[arg-type]


def test_router_route_with_reason_marks_exact_match(router: Router) -> None:
    team, exact = router.route_with_reason("Bug Report")
    assert team == "engineering-team"
    assert exact is True


def test_router_route_with_reason_marks_fallback(router: Router) -> None:
    team, exact = router.route_with_reason("unknown")
    assert team == DEFAULT_TEAM
    assert exact is False


def test_router_route_with_reason_non_string() -> None:
    r = Router()
    team, exact = r.route_with_reason(None)  # type: ignore[arg-type]
    assert team == DEFAULT_TEAM
    assert exact is False


# ---------------------------------------------------------------------------
# Sanity: the locked mapping really covers every CategoryClassifier class
# ---------------------------------------------------------------------------


def test_router_acknowledges_every_category_classifier_class() -> None:
    """``CategoryClassifier.CATEGORIES`` is the source of truth for labels;
    the router must have an entry for every one of them."""
    from ticket_router.models.category_classifier import CATEGORIES

    for category in CATEGORIES:
        assert category in CATEGORY_TO_TEAM, (
            f"Router is missing a team for category {category!r}"
        )
