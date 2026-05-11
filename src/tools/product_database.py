"""
product_database.py
A clean interface to the Heidstar product catalog.

Loads the catalog from data/heidstar_products.json and provides
methods agents can use to look up products, find matches, etc.

Why a tool? Agents should not parse JSON or know file paths —
they should just ask: "What products fit a stem cell research lab?"
"""

import json
from pathlib import Path


class ProductDatabaseTool:
    """
    Provides agents with a clean way to query the Heidstar product catalog.

    Usage:
        db = ProductDatabaseTool()
        all_products = db.get_all_products()
        flagship = db.get_product_by_id("HDS-MSCAN-200A")
        matches = db.find_products_for_signals(["FISH testing", "cancer research"])
    """

    def __init__(self):
        # Find the catalog file relative to this file's location.
        # Path(__file__) = full path to this Python file
        # .parent = the tools/ folder
        # .parent.parent = the src/ folder
        # .parent.parent.parent = the project root
        # / "data" / "heidstar_products.json" = our catalog file
        catalog_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "heidstar_products.json"
        )

        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Product catalog not found at {catalog_path}"
            )

        with open(catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

    # ─────────────────────────────────────────────────────
    # Company-level queries
    # ─────────────────────────────────────────────────────

    def get_company_info(self) -> dict:
        """Return Heidstar company info (credentials, location, etc.)."""
        return self.catalog["company"]

    def get_credentials(self) -> list[str]:
        """Return list of Heidstar's certifications / credentials."""
        return self.catalog["company"]["credentials"]

    # ─────────────────────────────────────────────────────
    # Product-level queries
    # ─────────────────────────────────────────────────────

    def get_all_products(self) -> list[dict]:
        """Return all products in the catalog."""
        return self.catalog["products"]

    def get_product_by_id(self, product_id: str) -> dict | None:
        """
        Look up a single product by its ID (e.g., 'HDS-MSCAN-200A').
        Returns None if not found.
        """
        for product in self.catalog["products"]:
            if product["id"] == product_id:
                return product
        return None

    def get_products_by_category(self, category: str) -> list[dict]:
        """
        Get all products in a category (e.g., 'Digital Pathology').
        Case-insensitive match.
        """
        category_lower = category.lower()
        return [
            p for p in self.catalog["products"]
            if p["category"].lower() == category_lower
        ]

    def find_products_for_signals(self, signals: list[str]) -> list[dict]:
        """
        Find products whose ideal_lead_signals overlap with the given signals.

        Example:
            signals = ["FISH testing", "cancer genetics research"]
            -> returns HDS-MSCAN-60F (matches both signals)

        This is the core matching logic the Qualifier Agent will use.

        Args:
            signals: List of phrases describing what a lead does.

        Returns:
            List of matching products, sorted by number of signal matches (best first).
        """
        signals_lower = [s.lower() for s in signals]
        matches = []

        for product in self.catalog["products"]:
            product_signals = [s.lower() for s in product["ideal_lead_signals"]]
            # Count how many of the lead's signals overlap with this product's signals
            match_count = sum(
                1 for lead_sig in signals_lower
                for prod_sig in product_signals
                if lead_sig in prod_sig or prod_sig in lead_sig
            )
            if match_count > 0:
                matches.append({
                    "product": product,
                    "match_count": match_count,
                })

        # Sort by best match first
        matches.sort(key=lambda m: m["match_count"], reverse=True)
        return [m["product"] for m in matches]

    # ─────────────────────────────────────────────────────
    # Future market queries
    # ─────────────────────────────────────────────────────

    def get_future_markets(self) -> list[dict]:
        """Return Heidstar's stated future market expansion plans."""
        return self.catalog["future_markets"]


# ─────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing ProductDatabaseTool...\n")
    db = ProductDatabaseTool()

    # Test 1: company info
    print("─" * 60)
    print("TEST 1 — Company info")
    print("─" * 60)
    company = db.get_company_info()
    print(f"Company: {company['name']} ({company['name_chinese']})")
    print(f"Founded: {company['founded']}, Location: {company['location']}")
    print(f"Credentials: {len(company['credentials'])} listed")
    print()

    # Test 2: list all products
    print("─" * 60)
    print("TEST 2 — All products")
    print("─" * 60)
    products = db.get_all_products()
    print(f"Found {len(products)} products in catalog:")
    for p in products:
        print(f"  • {p['id']} — {p['category']} ({p['tier']})")
    print()

    # Test 3: lookup by ID
    print("─" * 60)
    print("TEST 3 — Lookup HDS-MSCAN-200A")
    print("─" * 60)
    flagship = db.get_product_by_id("HDS-MSCAN-200A")
    if flagship:
        print(f"Found: {flagship['id']}")
        print(f"Category: {flagship['category']}")
        print(f"Description: {flagship['description'][:100]}...")
    print()

    # Test 4: signal matching (this is what Qualifier Agent will use)
    print("─" * 60)
    print("TEST 4 — Find products for a stem cell research lab")
    print("─" * 60)
    signals = [
        "stem cell laboratory",
        "iPSC research",
        "live cell imaging",
    ]
    matches = db.find_products_for_signals(signals)
    print(f"Lead signals: {signals}")
    print(f"Found {len(matches)} matching products:")
    for p in matches:
        print(f"  • {p['id']} — {p['description'][:80]}...")
    print()

    print("✅ All tests passed.")