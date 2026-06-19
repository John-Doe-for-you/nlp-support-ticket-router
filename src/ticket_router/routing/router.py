"""Category-to-team routing table. Implemented on Day 11."""

CATEGORY_TO_TEAM: dict[str, str] = {
    "Billing": "billing-team",
    "Authentication": "identity-team",
    "Bug Report": "engineering-team",
    "Feature Request": "product-team",
    "Technical Setup": "support-team",
}


def route_ticket(category: str) -> str:
    """Return the team responsible for the given category.

    Falls back to `support-team` for unknown categories so the API never 500s
    on novel input.
    """
    return CATEGORY_TO_TEAM.get(category, "support-team")
